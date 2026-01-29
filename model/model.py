from typing import Tuple,Optional,List,Union
from transformers import  PretrainedConfig,PreTrainedModel,GenerationMixin
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):

    def __init__(self, dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    freqs,attn_factor=1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim)),1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    def rotate_half(x:torch.Tensor):
        x1=x[..., :x.shape[-1]//2]
        x2=x[..., x.shape[-1]//2:]
        return torch.stack((-x2,x1),dim=-1).flatten(-2)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
        bs,slen,n_key_value_heads,head_dim=x.shape
        if n_rep==1:
            return x
        
        return x[:, :, :, None, :].expand(bs,slen,n_key_value_heads,n_rep,head_dim).reshape(bs,slen,n_key_value_heads*n_rep,head_dim)

class Attention(nn.Module):     #GQA
    def __init__(self,args:MiniMindConfig):
        super().__init__()

        self.num_key_value_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads%self.num_key_value_heads==0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads=args.num_attention_heads
        self.n_local_kv_heads=self.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_kv_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)     # Query投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Key投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Value投影
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)     # 输出投影
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(self,x:torch.Tensor,
                position_embeddings:Tuple[torch.Tensor,torch.Tensor],
                past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                use_cache=False,
                attention_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        bsz,seq_len,_=x.shape

        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)

        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)

        cos,sin=position_embeddings

        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq=xq.transpose(1,2)  # bsz, n_local_heads, seq_len, head_dim
        xk=repeat_kv(xk,self.n_rep).transpose(1,2)  # bsz, n_local_heads, seq_len, head_dim
        xv=repeat_kv(xv,self.n_rep).transpose(1,2)  # bsz, n_local_heads, seq_len, head_dim

        if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask == 1)):
            output=F.scaled_dot_product_attention(xq,xk,xv,dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            scores[:,:,:,-seq_len:]+=torch.triu(torch.full((seq_len,seq_len),float('-inf'),device=scores.device),diagonal=1)

            if attention_mask is not None:
                extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask=(1.0-extended_attention_mask)*-1e9
                scores=scores+extended_attention_mask
            
            scores=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=scores@xv
        
        output=output.transpose(1,2).reshape(bsz,seq_len,-1) # bsz, seq_len, n_local_heads*head_dim
        output=self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size=int(config.hidden_size*8/3)
            config.intermediate_size=64*((intermediate_size+63)//64)
        self.gate_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.down_proj=nn.Linear(config.intermediate_size,config.hidden_size,bias=False)
        self.up_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.dropout=nn.Dropout(config.dropout)
        self.act_fn=ACT2FN(config.hidden_act)
        

    def forward(self,x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MiniMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:MiniMindConfig):
        super().__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=config.hidden_size//config.num_attention_heads
        self.self_attn=Attention(config)

        self.layer_id=layer_id
        self.input_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=FeedForward(config) #if not config.use_moe else MoEFeedForward(config)

    def forward(self,hidden_states,postion_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        residual=hidden_states
        hidden_states,present_key_value=self.self_attn(
            self.input_layernorm(hidden_states),postion_embeddings,
            past_key_value,use_cache,attention_mask
        )
        hidden_states+=residual
        hidden_states=hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_value

class MiniMindModel(nn.Module):
    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config=config
        self.vocab_size=config.vocab_size
        self.num_hidden_layers=config.num_hidden_layers
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)
        self.layers=nn.ModuleList([MiniMindBlock(i,config) for i in range(config.num_hidden_layers)])
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        freqs_cos,freqs_sin=precompute_freqs_cis(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos",freqs_cos,False)
        self.register_buffer("freqs_sin",freqs_sin,False)
    
    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache:bool=False,
                **kwargs,                
                ):
        batch_size,seq_len=input_ids.shape

        if hasattr(past_key_values,'layers'):
            past_key_values=None
        past_key_values=past_key_values or [None]*len(self.layers)
        start_pos=past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states=self.dropout(self.embed_tokens(input_ids))

        position_embeddings=(self.freqs_cos[start_pos:start_pos+seq_len],self.freqs_sin[start_pos:start_pos+seq_len])
        
        presents=[]
        for layer_idx,(layer,past_key_values) in enumerate(zip(self.layers,past_key_values)):
            hidden_states,present=layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        hidden_states=self.norm(hidden_states)

        # aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self,config:MiniMindConfig):
        self.config=config or MiniMindConfig()
        super().__init__(self.config)
        self.model=MiniMindModel(self.config)
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        #输出层和嵌入层权重共享
        self.model.embed_tokens.weight=self.lm_head.weight

    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                labels:Optional[torch.Tensor]=None,
                past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep:Union[int,torch.Tensor]=0,
                **args):
        
        hidden_states,past_key_values=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        aux_loss = None     #后期可补充MoE

        slice_indices=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int)  else logits_to_keep 
        logits=self.lm_head(hidden_states)[:,slice_indices,:]

        loss=None
        if labels is not None:
            shifted_logits=logits[...,:-1,:].contiguous()
            shifted_labels=labels[...,1:].contiguous()
            loss=F.cross_entropy(shifted_logits.view(-1,shifted_logits.size(-1)),shifted_labels.view(-1),ignore_index=-100)

        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
