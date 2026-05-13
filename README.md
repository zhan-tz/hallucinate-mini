# hallucinate-mini

一个 5M 参数的语言模型在 MacBook 上 12 分钟训练完成，然后开始一本正经地胡说八道。

基于 [MiniMind](https://github.com/jingyaogong/minimind) 架构，压缩到 256 维 / 4 层 Transformer，用于学习 Transformer 语言模型的训练流程。

> ⚠️ 本项目是一个教学实验。模型输出不可用于任何严肃场景。

---

### 它说了些什么

**问：写一首关于秋天的五言诗。**

> 秋天，
> 一花，这是你有一些创意。
> 在夜晚的
> 风吹风，
> 这首诗真的花朵，

**问：请解释什么是光合作用。**

> 好的，我可以为您生成一篇关于环保的文章，我将为您写一篇关于环保的文章。

**问：推荐一些中国的美食。**

> 最近我正在学习，您想了解。我，您想象的天气。您好！我是"你"，"今天天气怎么样？"

**问：用 Python 写一个斐波那契数列。**

> I
>     else: 计算它们之间的关系，并非法
>     ### 计算出答案：
>     ### 1. 计算总部，和答案是：

它掌握了对话的格式和中文的语法结构，但词语之间的连接是错的。像一个学会了所有句式但没背过任何课文的学生去参加考试——每句话单独看都通顺，连起来就不知道在说什么了。

---

### 参数规模

| 模型 | 参数量 |
|------|--------|
| GPT-4 | ~1.8T |
| LLaMA-7B | 7B |
| MiniMind-3 | 64M |
| GPT-2 Small | 117M |
| **hallucinate-mini** | **4.98M** |

4.98M，大约是 GPT-2 Small 的 1/23。

### 架构

```
Token IDs → Embedding(6400×256) → [TransformerBlock × 4] → RMSNorm → lm_head
                                                  ↓
                            RMSNorm → GQA Attention(4Q/2KV) + Residual
                            RMSNorm → SwiGLU FFN           + Residual
```

与 MiniMind-3 相同的设计：RoPE 旋转位置编码、GQA 分组查询注意力、SwiGLU 激活、RMSNorm、权重绑定。只是维度和层数更小。

| 参数 | hallucinate-mini | MiniMind-3 |
|------|-----------------|------------|
| d_model | 256 | 768 |
| n_layers | 4 | 8 |
| n_heads / kv_heads | 4 / 2 | 8 / 4 |
| FFN intermediate | 832 | 2400 |
| vocab_size | 6400 | 6400 |

参数分布：FFN 51.3%，Embedding 32.9%，Attention 15.8%，Norm ~0%。

### 训练

MacBook (Apple Silicon, MPS)，FP32，~12 分钟完成全部训练。

**Pretrain**: 50k 条中文文本, 2 epochs, ~7 min

```
Step 1000: Loss = 14.97
Step 2000: Loss = 6.15
Step 6250: Loss = 5.04
```

**SFT**: 20k 条对话数据, 3 epochs, ~5 min

```
Step 1250: Loss = 5.60
Step 3750: Loss = 5.51
```

### 推理

- MPS, FP32, KV-cache ON: **50 tokens/s**
- MPS, FP32, KV-cache OFF: **20 tokens/s**

### 复现

```bash
cd hallucinate-mini

# Pretrain
python train_pretrain.py --epochs 2 --batch_size 16 --data_path ./data/pretrain_50k.jsonl

# SFT
python train_sft.py --epochs 3 --batch_size 16 --data_path ./data/sft_20k.jsonl

# 生成
python generate.py --mode auto --weight full_sft
```

### 文件结构

```
hallucinate-mini/
├── model.py              # 模型架构 (纯 PyTorch)
├── train_pretrain.py     # 预训练
├── train_sft.py          # SFT 微调
├── generate.py           # 推理
├── EXPERIMENT_NOTES.md   # 实验记录
├── data/                 # 训练数据子集
└── out/                  # 模型权重
```

### 致谢

基于 [MiniMind](https://github.com/jingyaogong/minimind) 的架构设计和训练数据。

### License

Apache 2.0
