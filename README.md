1. 使用RMSNorm归一化，不做均值中心化，节省开销同时效果不错
2. RoPE & YaRN，RoPE二维一组位置编码，YaRN上下文扩展
3. GQA，4个Q共享一个K和V，节约显存和算力