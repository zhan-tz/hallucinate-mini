# TinyMind 复现实验记录

> 从零实现并训练一个 ~5M 参数的微型 GPT（Transformer Decoder-Only）模型

---

## 一、实验目的

基于 MiniMind (https://github.com/jingyaogong/minimind) 的架构设计，从零使用纯 PyTorch 实现一个超小型 GPT 模型 TinyMind，完成 Pretrain → SFT 全流程训练，理解 Transformer 语言模型的核心组件和训练流程。

---

## 二、模型架构

### 2.1 整体结构

TinyMind 采用 **Transformer Decoder-Only** 架构，与 MiniMind-3 / Qwen3 / LLaMA 系列保持一致的核心设计：

```
Input Token IDs
    ↓
Token Embedding (6400 × 256)
    ↓
┌─────────────────────────────────┐
│  Transformer Block × 4          │
│  ┌───────────────────────────┐  │
│  │ RMSNorm → GQA Attention  │  │
│  │ + Residual Connection     │  │
│  │ RMSNorm → SwiGLU FFN     │  │
│  │ + Residual Connection     │  │
│  └───────────────────────────┘  │
└─────────────────────────────────┘
    ↓
Final RMSNorm
    ↓
lm_head (256 × 6400, 与 Embedding 权重共享)
    ↓
Output Logits → Next Token Prediction
```

### 2.2 配置参数对比

| 参数 | TinyMind | MiniMind-3 | GPT-2 Small |
|------|----------|------------|-------------|
| d_model (隐藏维度) | 256 | 768 | 768 |
| n_layers (层数) | 4 | 8 | 12 |
| n_heads (Q头数) | 4 | 8 | 12 |
| kv_heads (KV头数) | 2 | 4 | 12 (无GQA) |
| head_dim (每头维度) | 64 | 96 | 64 |
| intermediate_size (FFN) | 832 | 2400 | 3072 |
| vocab_size (词表大小) | 6400 | 6400 | 50257 |
| max_seq_len (最大长度) | 256 | 32768 | 1024 |
| rope_theta | 1e6 | 1e6 | — (使用绝对位置编码) |
| **总参数量** | **4.98M** | **64M** | **117M** |

### 2.3 参数量分布

| 模块 | 参数量 | 占比 |
|------|--------|------|
| Token Embedding (6400×256) | 1,638,400 (1.64M) | 32.9% |
| Attention (4层合计) | 786,944 (0.79M) | 15.8% |
| FFN SwiGLU (4层合计) | 2,555,904 (2.56M) | 51.3% |
| RMSNorm (4层×2 + final) | 2,304 (0.002M) | 0.0% |
| lm_head | 1,638,400 (与Embedding共享) | — |
| **总计** | **4,983,552 (4.98M)** | 100% |

每层 TransformerBlock 的参数明细（共 4 层，每层相同）：

| 子模块 | 参数量 |
|--------|--------|
| q_proj (256→256, no bias) | 65,536 |
| k_proj (256→128, no bias) | 32,768 |
| v_proj (256→128, no bias) | 32,768 |
| o_proj (256→256, no bias) | 65,536 |
| q_norm (RMSNorm, dim=64) | 64 |
| k_norm (RMSNorm, dim=64) | 64 |
| Attention 小计 | 196,736 |
| gate_proj (256→832, no bias) | 212,992 |
| up_proj (256→832, no bias) | 212,992 |
| down_proj (832→256, no bias) | 212,992 |
| FFN 小计 | 638,976 |
| input_layernorm + post_attention_layernorm | 512 |
| **每层合计** | **836,224** |

### 2.4 核心组件说明

#### (1) RMSNorm (Root Mean Square Layer Normalization)
- 与 LayerNorm 相比不需要计算均值，仅用均方根归一化
- 公式: `RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ`
- 被 LLaMA、Qwen、MiniMind 等现代 LLM 广泛使用

#### (2) RoPE (Rotary Position Embedding)
- 通过旋转变换将位置信息注入 Q/K 向量，使注意力天然感知相对位置
- 预计算 cos/sin 频率表: `freq = 1 / (θ^(2i/d))`, `θ = 1e6`
- 旋转变换: `q_rot = q * cos + rotate_half(q) * sin`

#### (3) GQA (Grouped Query Attention)
- 4 个 Query 头共享 2 组 Key/Value 头 (n_rep=2)
- 相比 MHA 减少了 KV-cache 大小，相比 MQA 保留了更多表达能力
- QK 归一化 (q_norm, k_norm) 提升训练稳定性

#### (4) SwiGLU FFN
- 门控机制: `FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))`
- 相比 ReLU/GeLU FFN 效果更好，已替代传统 FFN 成为标准
- intermediate_size = ceil(256 * π / 64) * 64 = 832

#### (5) 权重绑定 (Weight Tying)
- lm_head 与 Token Embedding 共享权重
- 减少约 1.64M 参数 (从 6.62M 降至 4.98M)
- GPT-2 以来的常用技巧

---

## 三、训练配置

### 3.1 硬件环境

- 设备: MacBook (Apple Silicon, MPS 加速)
- 训练精度: FP32 (MPS 不支持混合精度训练)
- 模型文件大小: 19.0 MB (FP32)

### 3.2 训练数据

| 阶段 | 数据集 | 样本数 | 来源 |
|------|--------|--------|------|
| Pretrain | pretrain_50k.jsonl | 50,000 | 从 MiniMind pretrain_t2t_mini.jsonl (127万条) 随机采样 |
| SFT | sft_20k.jsonl | 20,000 | 从 MiniMind sft_t2t_mini.jsonl (90万条) 随机采样 |

Pretrain 数据格式: `{"text": "..."}`
SFT 数据格式: `{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

### 3.3 超参数

#### Pretrain 阶段

| 参数 | 值 |
|------|-----|
| epochs | 2 |
| batch_size | 16 |
| accumulation_steps | 2 |
| 有效 batch_size | 32 |
| learning_rate | 5e-4 |
| LR Schedule | 余弦衰减 (0.1*lr ~ lr) |
| max_seq_len | 128 |
| optimizer | AdamW |
| grad_clip | 1.0 |
| 总训练步数 | 6,250 |

#### SFT 阶段

| 参数 | 值 |
|------|-----|
| epochs | 3 |
| batch_size | 16 |
| accumulation_steps | 2 |
| 有效 batch_size | 32 |
| learning_rate | 1e-5 |
| LR Schedule | 余弦衰减 (0.1*lr ~ lr) |
| max_seq_len | 256 |
| optimizer | AdamW |
| grad_clip | 1.0 |
| 基础权重 | pretrain_256.pth |
| 总训练步数 | 3,750 |

---

## 四、训练结果

### 4.1 Pretrain Loss 曲线

```
Step    1000: Loss = 14.97    (LR: 4.72e-04)
Step    2000: Loss = 6.15     (LR: 3.96e-04)
Step    3000: Loss = 5.64     (LR: 2.89e-04)
Step    3125: Loss = 5.48     (LR: 2.75e-04)  ← Epoch 1 结束, 平均 Loss: 8.78
Step    4125: Loss = 5.31     (LR: 1.67e-04)
Step    5125: Loss = 5.17     (LR: 8.50e-05)
Step    6125: Loss = 5.10     (LR: 5.04e-05)
Step    6250: Loss = 5.04     (LR: 5.00e-05)  ← Epoch 2 结束, 平均 Loss: 5.18
```

Loss 从 ~245 (初始) 快速下降至 ~5.0，模型学会了基本的中文语言模式。
- Epoch 1 平均 Loss: 8.78
- Epoch 2 平均 Loss: 5.18

### 4.2 SFT Loss 曲线

```
Step     500: Loss = 5.85     (LR: 9.61e-06)  ← Epoch 1
Step    1000: Loss = 5.64     (LR: 8.51e-06)
Step    1250: Loss = 5.60     (LR: 7.75e-06)  ← Epoch 1 结束, 平均 Loss: 5.71
Step    1750: Loss = 5.58     (LR: 5.97e-06)  ← Epoch 2
Step    2250: Loss = 5.54     (LR: 4.11e-06)
Step    2500: Loss = 5.53     (LR: 3.25e-06)  ← Epoch 2 结束, 平均 Loss: 5.55
Step    3000: Loss = 5.51     (LR: 1.86e-06)  ← Epoch 3
Step    3500: Loss = 5.53     (LR: 1.10e-06)
Step    3750: Loss = 5.51     (LR: 1.00e-06)  ← Epoch 3 结束, 平均 Loss: 5.52
```

SFT Loss 从 5.85 缓慢下降至 5.51，收敛趋于平稳。

### 4.3 训练耗时

| 阶段 | 总步数 | 耗时 |
|------|--------|------|
| Pretrain (2 epochs) | 6,250 步 | ~7 分钟 |
| SFT (3 epochs) | 3,750 步 | ~5 分钟 |
| **合计** | **10,000 步** | **~12 分钟** |

### 4.4 推理速度 (Apple Silicon MPS, FP32)

| 模式 | 速度 |
|------|------|
| KV-cache ON | **50.0 tokens/s** |
| KV-cache OFF | **19.8 tokens/s** |
| KV-cache 加速比 | **2.5x** |

---

## 五、生成效果展示

### 5.1 Pretrain 基座模型 (文本续写)

```
Prompt: 今天天气真不错
Output: 今天天气真不错的天气预报，我建议您想象力和"你"的天气预报。
        好的，我来帮您，让我在的
```

预训练模型能生成通顺的中文语句，但内容缺乏针对性和逻辑性。

### 5.2 SFT 微调后模型 (对话模式)

```
User: 请用Python写一个斐波那契数列的计算函数。
Model: I
    else: 计算它们之间的关系，并非法
    ### 计算出答案：
    ### 1. 计算总部，和答案是：
    ### 计算，用于计算裁症
    ### 1. 计算
    else
    ### 1. 计算它们的后，在数字化，计算它们的和。
    它是中，用于计算方法
```

```
User: 请解释什么是光合作用。
Model: 好的，我是
    好的，我可以为您生成一篇关于环保的文章，
    我将为您写一篇关于环保的文章。
```

```
User: 推荐一些中国的美食。
Model: 最近我正在学习，您想了解。我，您想象的天气。
       您好！我是"你"，"今天天气怎么样？"和"我"或"我"可能表达的天气吗？"
       或"我来帮助，建议"。
```

```
User: 写一首关于秋天的五言诗。
Model: 秋天，
       一花，这是你有一些创意。
       在夜晚的
       风吹风，
       这首诗真的花朵，
       在这首歌词。
```

### 5.3 效果分析

- **优点**: 模型学会了基本的中文语法结构、标点使用、部分常见词汇搭配
- **局限**: 
  - 参数量仅 4.98M，知识容量极其有限
  - 训练数据量少（50k pretrain + 20k SFT），远不足以学到可靠的事实知识
  - 生成内容存在明显的幻觉和重复
  - 代码生成、逻辑推理等能力基本不具备
- **对比 MiniMind-3 (64M)**: TinyMind 的参数量仅为 MiniMind-3 的 1/13，效果差距是合理的

---

## 六、关键代码实现要点

### 6.1 从零实现的组件清单

所有代码在 `tinymind/model.py` 中，共约 600 行，不依赖 transformers 库：

1. `RMSNorm` - 14 行
2. `precompute_freqs_cis()` + `apply_rotary_pos_emb()` - RoPE 位置编码
3. `repeat_kv()` - GQA 的 KV 头扩展
4. `Attention` - 多头注意力 + KV-cache
5. `FeedForward` - SwiGLU 前馈网络
6. `TransformerBlock` - Pre-Norm 残差块
7. `TinyMindModel` - 基座模型
8. `TinyMindForCausalLM` - 因果语言模型 + generate()

### 6.2 KV-cache 加速原理

```
无 cache: 每步都重新计算所有 token 的 attention → O(n²) 每步
有 cache: 只计算新 token，拼接历史 KV → O(n) 每步

实测加速比: 2.5x (100 token 生成)
```

### 6.3 权重绑定 (Weight Tying)

```python
# lm_head 和 embed_tokens 共享同一份权重
self.lm_head.weight = self.model.embed_tokens.weight
# 节省 1.64M 参数 (从 6.62M → 4.98M)
```

---

## 七、文件清单

```
tinymind/
├── model.py                  # 模型架构 (~600行, 纯 PyTorch)
├── train_pretrain.py         # 预训练脚本
├── train_sft.py              # SFT 微调脚本
├── generate.py               # 推理/对话脚本
├── tokenizer.json            # 分词器 (6400 词表)
├── tokenizer_config.json
├── data/
│   ├── pretrain_50k.jsonl    # 预训练数据 (50,000 条)
│   └── sft_20k.jsonl         # SFT 数据 (20,000 条)
└── out/
    ├── pretrain_256.pth      # 预训练权重 (19 MB)
    └── full_sft_256.pth      # SFT 权重 (19 MB)
```

---

## 八、运行命令

```bash
cd tinymind

# 步骤1: 预训练 (~7 min on MPS)
python train_pretrain.py \
  --epochs 2 --batch_size 16 --max_seq_len 128 \
  --data_path ./data/pretrain_50k.jsonl

# 步骤2: SFT 微调 (~5 min on MPS)
python train_sft.py \
  --epochs 3 --batch_size 16 --max_seq_len 256 \
  --data_path ./data/sft_20k.jsonl

# 步骤3: 自动测试
python generate.py --mode auto --weight full_sft

# 步骤4: 交互对话
python generate.py --mode chat --weight full_sft
```

---

## 九、参考资料

1. MiniMind: https://github.com/jingyaogong/minimind
2. LLaMA: Touvron et al., "LLaMA: Open and Efficient Foundation Language Models", 2023
3. RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
4. SwiGLU: Shazeer, "GLU Variants Improve Transformer", 2020
5. GQA: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023
6. RMSNorm: Zhang & Sennrich, "Root Mean Square Layer Normalization", 2019
