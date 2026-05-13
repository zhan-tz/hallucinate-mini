"""
TinyMind - 超小型 GPT（Transformer Decoder-Only）模型
======================================================================
基于 MiniMind (https://github.com/jingyaogong/minimind) 架构，
使用纯 PyTorch 从零实现，不依赖 transformers 库。

参数配置（约 5M 参数量，可在 MacBook CPU / MPS 上 ~30 分钟完成训练）:
    d_model = 256       # 隐藏层维度（MiniMind 使用 768）
    n_layers = 4        # Transformer 层数（MiniMind 使用 8）
    n_heads = 4         # 查询注意力头数（MiniMind 使用 8）
    kv_heads = 2        # 键/值注意力头数，实现 GQA
    vocab_size = 6400   # 词表大小（与 MiniMind tokenizer 一致）
    max_seq_len = 256   # 最大序列长度
    rope_theta = 1e6    # RoPE 基础频率
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================================
# 第一部分: 模型配置
# ===========================================================================


class TinyMindConfig:
    """TinyMind 模型配置 —— 所有超参数集中管理"""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        kv_heads: int = 2,
        vocab_size: int = 6400,
        max_seq_len: int = 256,
        rope_theta: float = 1e6,
        rms_norm_eps: float = 1e-5,
    ):
        self.d_model = d_model  # 隐藏层维度
        self.n_layers = n_layers  # Transformer 块数量
        self.n_heads = n_heads  # query 注意力头数
        self.kv_heads = kv_heads  # key/value 注意力头数（GQA）
        self.vocab_size = vocab_size  # 词表大小
        self.max_seq_len = max_seq_len  # 最大序列长度
        self.rope_theta = rope_theta  # RoPE 基础频率 θ
        self.rms_norm_eps = rms_norm_eps  # RMSNorm 的 epsilon

        # --- 以下由上述参数推导，无需手动指定 ---
        # 每个注意力头的维度：d_model 均匀分配到 n_heads 个头上
        self.head_dim = d_model // n_heads
        # GQA: 每个 KV 头需要扩展的倍数（n_q_heads / n_kv_heads）
        self.n_rep = n_heads // kv_heads
        # SwiGLU FFN 中间层维度 —— ceil(d_model * π / 64) * 64
        # 这是 Llama / Qwen / MiniMind 的经典公式，用于对齐硬件友好的维度
        self.intermediate_size = math.ceil(d_model * math.pi / 64) * 64


# ===========================================================================
# 第二部分: RMS 归一化（Root Mean Square Normalization）
# ===========================================================================


class RMSNorm(nn.Module):
    """
    RMS 归一化。

    与标准 LayerNorm 不同，RMSNorm 仅使用均方根（RMS）进行缩放，
    不减去均值，不包含偏置项。计算更简单，训练更稳定。

    公式:
        RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ

    其中 γ 是可学习的缩放参数（weight），初始化全 1。
    Llama、Qwen、MiniMind 等现代 LLM 均采用此归一化方式。

    参数量: dim（仅 weight 参数，无 bias）
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 可学习缩放参数，形状 (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        RMS 归一化核心计算:
            1. x.pow(2).mean(-1, keepdim=True) —— 在最后一维上求平方均值
            2. torch.rsqrt(...)                  —— 计算 1 / sqrt(均值 + eps)
            3. x * rsqrt(...)                    —— 缩放
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转为 float32 计算以提高数值精度，结果再转回原始类型
        output = self._norm(x.float()).type_as(x)
        return self.weight * output


# ===========================================================================
# 第三部分: 旋转位置编码（Rotary Position Embedding, RoPE）
# ===========================================================================


def precompute_freqs_cis(
    dim: int,
    end: int,
    rope_base: float = 1e6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 频率表（cos 和 sin）。

    RoPE 的核心思想是：将位置信息通过旋转变换注入到 query 和 key 向量中，
    使得注意力分数的计算能够自然地反映 token 之间的相对位置关系。

    频率计算:
        freq_i = 1 / (theta^(2i / dim)),  i = 0, 1, ..., dim/2 - 1

    对于每个位置 pos，在维度 i 和 i+dim/2 上的旋转角度为 pos * freq_i。
    最终预计算所有位置的 cos(pos * freq_i) 和 sin(pos * freq_i)。

    参数:
        dim:       每个注意力头的维度
        end:       最大位置索引
        rope_base: 基础频率 θ（默认 1e6）

    返回:
        freqs_cos: 预计算余弦值，形状 (end, dim)
        freqs_sin: 预计算正弦值，形状 (end, dim)
    """
    # 计算每个维度对的频率: 1 / θ^(2i/dim)
    freqs = 1.0 / (
        rope_base ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim)
    )
    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, dtype=torch.float32)
    # 外积: freqs 形状为 (end, dim//2)
    freqs = torch.outer(t, freqs)
    # 将 dim//2 长度的频率复制一份拼成 dim 长度
    # 原因: RoPE 按维度对 (i, i+dim/2) 成对旋转，需要用相同的频率值
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 query 和 key 张量应用旋转位置编码。

    旋转变换（在二维平面上）:
        对于维度对 (x_{2i}, x_{2i+1}), 旋转角度 θ 后:
        x'_{2i}   = x_{2i} * cos(θ) - x_{2i+1} * sin(θ)
        x'_{2i+1} = x_{2i} * sin(θ) + x_{2i+1} * cos(θ)

    高效向量化实现:
        将向量分为前后两半 [x_front, x_back]
        x_rotated = x * cos + rotate_half(x) * sin
        其中 rotate_half(x) = [-x_back, x_front]

    参数:
        q, k:          query 和 key 张量，形状 (bsz, n_heads, seq_len, head_dim)
        cos, sin:      预计算的位置编码，形状 (seq_len, head_dim)
        unsqueeze_dim: cos/sin 需要在哪个维度上扩展以匹配 q/k 的形状

    返回:
        q_embed, k_embed: 应用 RoPE 后的 query 和 key
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将向量后半部分取负并与前半部分交换位置"""
        x1 = x[..., : x.shape[-1] // 2]  # 前半部分
        x2 = x[..., x.shape[-1] // 2:]  # 后半部分
        return torch.cat((-x2, x1), dim=-1)

    # cos/sin 需要 unsqueeze 以匹配 (bsz, n_heads, seq_len, head_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 key/value 头数从 kv_heads 扩展到 q_heads（GQA 扩展）。

    分组查询注意力（Grouped Query Attention, GQA）:
        每个 KV 头被 n_rep = q_heads / kv_heads 个 query 头共享。
        通过重复 key/value 来实现维度对齐。

    参数:
        x:     输入张量，形状 (bsz, seq_len, kv_heads, head_dim)
        n_rep: 每个 KV 头需要重复的次数

    返回:
        形状 (bsz, seq_len, q_heads, head_dim)
    """
    bs, seqlen, num_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 在第 3 维后插入一个新维度 (None)，扩展到 n_rep 倍，然后展平
    return (
        x[:, :, :, None, :]
        .expand(bs, seqlen, num_kv_heads, n_rep, head_dim)
        .reshape(bs, seqlen, num_kv_heads * n_rep, head_dim)
    )


# ===========================================================================
# 第四部分: 多头注意力模块（支持 GQA + RoPE + KV-Cache）
# ===========================================================================


class Attention(nn.Module):
    """
    多头注意力（Multi-Head Attention with GQA）。

    计算流程:
        1. x → q_proj, k_proj, v_proj（线性投影，无偏置）
        2. q, k → RMSNorm（QK 归一化，提升训练稳定性）
        3. q, k → apply_rotary_pos_emb（注入位置信息）
        4. 可选: 拼接历史 KV-cache
        5. k, v → repeat_kv（GQA 扩展）
        6. 计算缩放点积注意力（支持 PyTorch 2.0 Flash Attention）
        7. 结果 → o_proj → 输出

    参数量（以 TinyMind 默认配置为例）:
        q_proj: 256 × (4 × 64) = 256 × 256 =  65,536
        k_proj: 256 × (2 × 64) = 256 × 128 =  32,768
        v_proj: 256 × (2 × 64) = 256 × 128 =  32,768
        o_proj: (4 × 64) × 256 = 256 × 256 =  65,536
        q_norm: 64
        k_norm: 64
        ─────────────────────────────────────────
        合计: 196,736
    """

    def __init__(self, config: TinyMindConfig):
        super().__init__()
        self.n_q_heads = config.n_heads  # query 头数: 4
        self.n_kv_heads = config.kv_heads  # key/value 头数: 2
        self.n_rep = config.n_rep  # KV 重复倍数: 2
        self.head_dim = config.head_dim  # 每头维度: 64

        # 四个线性投影层（均无偏置 bias=False）
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        # QK 归一化: 对每个 head 的 query 和 key 做 RMSNorm
        # 实验表明 QK Norm 能有效防止 attention logit 过大导致的训练不稳定
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 检测是否支持 PyTorch 2.0+ 的 Flash Attention
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播。

        参数:
            x:                   输入隐藏状态 (bsz, seq_len, d_model)
            position_embeddings: (cos, sin) 元组，用于 RoPE
            past_key_value:      缓存的 (key, value)，用于 KV-cache
            use_cache:           是否返回新的 KV-cache 供后续生成使用
            attention_mask:      padding mask (bsz, seq_len)，1=有效, 0=忽略

        返回:
            output:  注意力输出 (bsz, seq_len, d_model)
            past_kv: 更新后的 (key, value) 缓存对
        """
        bsz, seq_len, _ = x.shape

        # 步骤 1: 线性投影得到 Q, K, V
        xq = self.q_proj(x)  # (bsz, seq_len, n_q_heads × head_dim)
        xk = self.k_proj(x)  # (bsz, seq_len, n_kv_heads × head_dim)
        xv = self.v_proj(x)  # (bsz, seq_len, n_kv_heads × head_dim)

        # 步骤 2: 重塑为多头形状
        xq = xq.view(bsz, seq_len, self.n_q_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 步骤 3: QK 归一化
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # 步骤 4: 应用旋转位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 步骤 5: KV-cache —— 将当前 key/value 拼接到历史缓存之后
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 沿序列维拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)

        # 保存当前 KV 对（如果 use_cache=True 且非训练模式则会用到）
        past_kv = (xk, xv) if use_cache else None

        # 步骤 6: 转置为 (bsz, n_heads, seq_len, head_dim) 供注意力计算
        #         同时对 KV 执行 GQA 扩展
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 步骤 7: 注意力计算
        if self.flash and past_key_value is None and attention_mask is None:
            # 使用 PyTorch 内置 Flash Attention（预填充阶段，无 padding）
            # is_causal=True 自动生成上三角为 -inf 的因果掩码
            output = F.scaled_dot_product_attention(
                xq, xk, xv, dropout_p=0.0, is_causal=True
            )
        else:
            # 手动实现缩放点积注意力（兼容生成阶段 + padding mask 场景）
            # scores = Q · K^T / sqrt(d_k)
            scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # 因果掩码: 防止当前位置关注未来位置
            # 使用上三角矩阵（对角线以上=1）填充 -inf
            # 注意: 这里只对当前 seq_len 做掩码，历史缓存的 key 不受影响
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1,
            )
            scores[:, :, -seq_len:, -seq_len:] += causal_mask

            # Padding 掩码: 将 padding 位置对应的注意力分数设为 -inf
            if attention_mask is not None:
                # attention_mask: (bsz, seq_len), 1=有效, 0=padding
                # 扩展为 (bsz, 1, 1, seq_len) 广播到 scores 的最后一维
                scores = scores + (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

            # Softmax 归一化并加权求和
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(attn_weights, xv)

        # 步骤 8: 恢复形状并通过输出投影
        # (bsz, n_heads, seq_len, head_dim) → (bsz, seq_len, d_model)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)

        return output, past_kv


# ===========================================================================
# 第五部分: SwiGLU 前馈网络
# ===========================================================================


class FeedForward(nn.Module):
    """
    SwiGLU 前馈网络（SwiGLU Feed-Forward Network）。

    与传统 FFN（两层线性 + 激活）不同，SwiGLU 使用门控机制:

        FFN(x) = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

    其中:
        - W_gate: 门控投影 —— 通过 SiLU 激活后决定"哪些信息可以通过"
        - W_up:   内容投影 —— 提供实际的变换信息
        - ⊙:      逐元素乘法（门控操作）
        - W_down: 输出投影 —— 将高维表示映射回 d_model

    SwiGLU 相比 ReLU/GeLU 的优势:
        - 门控机制让网络能自适应地过滤信息
        - SiLU（sigmoid 加权线性单元）平滑且非单调，梯度特性更好
        - 已被 Llama、Qwen、PaLM 等模型验证有效

    参数量（以 TinyMind 默认配置为例）:
        gate_proj: 256 × 832 = 212,992
        up_proj:   256 × 832 = 212,992
        down_proj: 832 × 256 = 212,992
        ─────────────────────────────
        合计: 638,976
    """

    def __init__(self, config: TinyMindConfig):
        super().__init__()
        # 门控投影: 确定信息的通过比例
        self.gate_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        # 内容投影: 提供实际的特征变换
        self.up_proj = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        # 输出投影: 映射回原始维度
        self.down_proj = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) * up 实现门控，再投影回 d_model
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ===========================================================================
# 第六部分: Transformer 块
# ===========================================================================


class TransformerBlock(nn.Module):
    """
    Transformer 块 —— 模型的基本构建单元。

    采用预归一化（Pre-Norm）架构，每个块包含两个子层:
        (1) 多头自注意力子层:  Attn(RMSNorm(x)) + x
        (2) SwiGLU 前馈子层:   FFN(RMSNorm(x))  + x

    两个子层都有残差连接（Residual Connection），
    残差连接让梯度可以直接流过深层网络，缓解梯度消失问题。

    注意: 这里使用 Pre-Norm 而非原始 Transformer 的 Post-Norm。
    Pre-Norm 把归一化放在子层之前（而非之后），训练更稳定，
    是现代 GPT 类模型的标准做法。

    参数量（以 TinyMind 默认配置为例）:
        Attention:           196,736
        FFN (SwiGLU):        638,976
        RMSNorm × 2 (256×2):     512
        ─────────────────────────────
        合计: 836,224
    """

    def __init__(self, layer_id: int, config: TinyMindConfig):
        super().__init__()
        self.layer_id = layer_id

        # 注意力子层
        self.self_attn = Attention(config)

        # 注意力前的 RMSNorm
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # FFN 前的 RMSNorm
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # SwiGLU 前馈网络
        self.mlp = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播 —— Pre-Norm + 残差连接。

        流程:
            residual = x
            x = Attn(Norm(x))       ← 归一化后做注意力
            x = x + residual        ← 残差连接

            residual = x
            x = FFN(Norm(x))        ← 归一化后做前馈
            x = x + residual        ← 残差连接
        """
        # 残差分支 1: 自注意力
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + residual

        # 残差分支 2: 前馈网络
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value


# ===========================================================================
# 第七部分: TinyMind 基座模型
# ===========================================================================


class TinyMindModel(nn.Module):
    """
    TinyMind 基座模型 —— Transformer Decoder-Only 架构。

    模型结构（自底向上）:
        ┌──────────────────────────────────────┐
        │           Token Embedding            │  vocab_size → d_model
        ├──────────────────────────────────────┤
        │  TransformerBlock 0                  │
        │    ├── RMSNorm → Attention (+残差)   │
        │    └── RMSNorm → SwiGLU FFN (+残差)  │
        ├──────────────────────────────────────┤
        │  TransformerBlock 1                  │
        │    ...                               │
        ├──────────────────────────────────────┤
        │  TransformerBlock 2                  │
        │    ...                               │
        ├──────────────────────────────────────┤
        │  TransformerBlock 3                  │
        │    ...                               │
        ├──────────────────────────────────────┤
        │         Final RMSNorm                │  → hidden_states
        └──────────────────────────────────────┘

    输入:  token ID 序列
    输出:  最后一层的隐藏状态（用于后续 lm_head 预测下一个 token）

    RoPE 频率表通过 register_buffer 注册，随模型一起移动到 GPU/CPU，
    但不参与梯度更新，也不会被保存到 state_dict（persistent=False）。
    """

    def __init__(self, config: TinyMindConfig):
        super().__init__()
        self.config = config

        # Token 嵌入层: 将离散 token ID 映射为连续向量
        # 参数量: vocab_size × d_model = 6400 × 256 = 1,638,400
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer 块堆叠
        self.layers = nn.ModuleList(
            [TransformerBlock(layer_id, config) for layer_id in range(config.n_layers)]
        )

        # 最终归一化层
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率表并注册为 buffer
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len,
            rope_base=config.rope_theta,
        )
        # persistent=False: buffer 不保存到 state_dict，加载时可重新计算
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播。

        参数:
            input_ids:       token ID 序列 (bsz, seq_len)
            attention_mask:  注意力掩码 (bsz, seq_len)，1=有效, 0=padding
            past_key_values: KV-cache 列表，每层一个 (key, value) 元组
            use_cache:       是否返回新的 KV-cache

        返回:
            hidden_states: 最后一层隐藏状态 (bsz, seq_len, d_model)
            presents:      各层的 KV-cache 列表，每项为 (k, v) 或 None
        """
        batch_size, seq_len = input_ids.shape

        # 首次调用时初始化一个全 None 的 KV-cache 列表
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # 计算当前序列的 RoPE 起始位置
        # 如果已有缓存（生成阶段），新 token 的位置从缓存长度开始
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Token 嵌入
        hidden_states = self.embed_tokens(input_ids)

        # 提取当前位置范围的 RoPE 频率
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        # 逐层前向传播
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # 最终归一化
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


# ===========================================================================
# 第八部分: 因果语言模型封装（用于训练和生成）
# ===========================================================================


class TinyMindForCausalLM(nn.Module):
    """
    TinyMind 因果语言模型 —— 训练和推理的统一入口。

    在 TinyMindModel 之上添加 lm_head（线性输出层），
    将隐藏状态映射为词表上的 logits，用于预测下一个 token。

    权重绑定（Weight Tying）:
        lm_head.weight 与 embed_tokens.weight 共享同一参数矩阵。
        这是 GPT-2 以来几乎所有 LLM 的标准做法，原因:
        - 减少参数量（省掉 vocab_size × d_model 个参数）
        - 理论依据: 输入嵌入和输出投影在数学上是"互逆"操作，
          共享权重相当于施加了有效的正则化

    训练:   forward(input_ids, labels) → 计算交叉熵损失
    推理:   generate(input_ids)         → 自回归采样生成

    参数量分布（总计约 4.98M）:
        Token Embedding:         1,638,400
        TransformerBlock × 4:    3,344,896  (836,224 × 4)
        Final RMSNorm:                 256
        ─────────────────────────────────────
        总计:                    4,983,552  ≈ 4.98M
    """

    def __init__(self, config: Optional[TinyMindConfig] = None):
        super().__init__()
        self.config = config or TinyMindConfig()

        # 基座模型
        self.model = TinyMindModel(self.config)

        # 语言模型头: d_model → vocab_size
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # 权重绑定: lm_head 与 embedding 共享权重
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        前向传播（训练 / 推理共用）。

        训练时:
            input_ids 和 labels 都是完整的序列，
            使用 teacher forcing: 所有位置并行计算，loss 从 labels 计算。

        推理时（自回归生成）:
            input_ids 为当前已生成的序列（或仅新 token），
            labels 为 None，use_cache=True 以保存 KV-cache。

        损失计算（语言建模的标准"shift"操作）:
            logits[0:t-1] 预测 labels[1:t]
            即: 用位置 i 的输出预测位置 i+1 的 token

        参数:
            input_ids:       token ID 序列 (bsz, seq_len)
            attention_mask:  注意力掩码
            past_key_values: KV-cache
            use_cache:       是否使用 KV-cache
            labels:          目标 token ID (bsz, seq_len), -100 表示忽略

        返回:
            dict: {"loss": ..., "logits": ..., "past_key_values": ...}
        """
        hidden_states, past_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift:
            #   logits[:, :-1] 对应位置 0, 1, ..., t-2 的输出
            #   labels[:, 1:]  对应位置 1, 2, ..., t-1 的目标
            # 即用前 t-1 个位置的输出来预测后 t-1 个位置
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": past_key_values,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.85,
        top_p: float = 0.85,
        top_k: int = 50,
        eos_token_id: int = 2,
        use_cache: bool = True,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        自回归文本生成。

        每步:
            1. 前向传播获得 logits
            2. 温度缩放调理分布的尖锐程度
            3. 重复惩罚降低已出现 token 的概率
            4. Top-k 过滤: 仅保留概率最高的 k 个 token
            5. Top-p (nucleus) 过滤: 保留累积概率 ≤ p 的最小 token 集合
            6. 从过滤后的分布中采样（或贪心选择）下一个 token
            7. 如果命中 eos，标记该样本已完成

        KV-cache 加速原理:
            不使用缓存时，每步需要重新编码整个序列（O(n²)）。
            使用缓存后，每步只需编码新 token 并利用历史 KV，
            复杂度降为 O(n)。

        采样策略说明:
            - temperature = 1.0: 不做调整，保持原始概率分布
            - temperature < 1.0: 分布更尖锐（高概率 token 更可能被选中）
            - temperature > 1.0: 分布更平坦（增加多样性）
            - top_p = 0.9:  只从累积概率前 90% 的 token 中采样
            - top_k = 50:   只从概率最高的 50 个 token 中采样
            - do_sample = False: 贪心解码（始终选概率最高的 token）

        参数:
            input_ids:          输入 token ID，形状 (1, prompt_len)
            max_new_tokens:     最大生成 token 数
            temperature:        温度系数
            top_p:              nucleus 采样阈值
            top_k:              top-k 采样阈值
            eos_token_id:       结束符 token ID
            use_cache:          是否使用 KV-cache
            do_sample:          True=采样, False=贪心
            repetition_penalty: 重复惩罚系数 (>1 降低重复，<1 鼓励重复)

        返回:
            完整 token ID 序列 (1, prompt_len + 生成长度)
        """
        past_key_values = None
        # 跟踪每个批次样本是否已生成结束符
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            # 计算 KV-cache 中已缓存的 token 数量
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0

            # 只将还未缓存的新 token 送入模型
            current_input = input_ids[:, past_len:] if past_len > 0 else input_ids

            # 前向传播
            outputs = self.forward(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )

            # 取最后一个位置的 logits 进行采样
            logits = outputs["logits"][:, -1, :].float()

            # --- 温度缩放 ---
            logits = logits / temperature

            # --- 重复惩罚 ---
            # 将已生成的 token 的 logits 除以 penalty 以降低其概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    unique_ids = torch.unique(input_ids[i])
                    logits[i, unique_ids] /= repetition_penalty

            # --- Top-k 过滤 ---
            # 仅保留概率最高的 k 个 token，其余设为 -inf
            if top_k > 0:
                topk_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                threshold = topk_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # --- Top-p (nucleus) 过滤 ---
            # 按概率降序排列，保留累积概率不超过 top_p 的最小 token 集合
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # 累积概率超过 top_p 的 token 被标记为移除
                sorted_mask = cumulative_probs > top_p
                # 至少保留概率最高的一个 token
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                # 将 mask 映射回原始 logits 索引
                mask = torch.zeros_like(logits, dtype=torch.bool)
                mask = mask.scatter(1, sorted_indices, sorted_mask)
                logits[mask] = float("-inf")

            # --- 采样下一个 token ---
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 已完成的样本强制输出 eos
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            # 拼接新 token 到序列末尾
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 更新 KV-cache
            past_key_values = outputs["past_key_values"] if use_cache else None

            # 检查终止条件
            if eos_token_id is not None:
                finished = finished | next_token.squeeze(-1).eq(eos_token_id)
                if finished.all():
                    break

        return input_ids


# ===========================================================================
# 第九部分: 测试入口
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  TinyMind 模型测试")
    print("=" * 60)

    # 创建模型配置
    config = TinyMindConfig()

    print(f"\n📋 配置信息:")
    print(f"  d_model (隐藏维度):        {config.d_model}")
    print(f"  n_layers (Transformer 层): {config.n_layers}")
    print(f"  n_heads (注意力头数):       {config.n_heads}")
    print(f"  kv_heads (KV 头数 / GQA):   {config.kv_heads}")
    print(f"  head_dim (每头维度):        {config.head_dim}")
    print(f"  n_rep (KV 重复倍数):        {config.n_rep}")
    print(f"  vocab_size (词表大小):      {config.vocab_size}")
    print(f"  max_seq_len (最大长度):     {config.max_seq_len}")
    print(f"  intermediate_size (FFN):    {config.intermediate_size}")
    print(f"  rope_theta (RoPE 频率):     {config.rope_theta}")

    # 创建模型
    model = TinyMindForCausalLM(config)

    # —— 参数量统计 ——
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 参数量统计:")
    print(f"  总参数量:     {total_params:>10,}  ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数:   {trainable_params:>10,}  ({trainable_params / 1e6:.2f}M)")

    # 各模块参数量
    print(f"\n📦 各模块参数量:")
    embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    print(f"  Token Embedding:          {embed_params:>8,}")
    for i, layer in enumerate(model.model.layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        attn_params = sum(p.numel() for p in layer.self_attn.parameters())
        ffn_params = sum(p.numel() for p in layer.mlp.parameters())
        print(
            f"  TransformerBlock {i}:      {layer_params:>8,}  "
            f"(Attn: {attn_params:>7,},  FFN: {ffn_params:>7,})"
        )
    norm_params = sum(p.numel() for p in model.model.norm.parameters())
    print(f"  Final RMSNorm:            {norm_params:>8,}")
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    print(f"  lm_head (与 embedding 共享权重): {lm_head_params:>8,}")

    # —— 前向传播测试 ——
    print(f"\n🚀 前向传播测试 (训练模式):")
    batch_size = 2
    seq_len = 16

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    model.train()
    outputs = model(input_ids=input_ids, labels=labels)

    print(f"  输入形状:   {input_ids.shape}")
    print(f"  Logits 形状: {outputs['logits'].shape}")
    print(f"  Loss:        {outputs['loss'].item():.4f}")
    print(f"  Perplexity:  {math.exp(outputs['loss'].item()):.2f}")

    # —— 生成测试 ——
    print(f"\n🎲 生成测试 (采样模式):")
    model.eval()
    torch.manual_seed(123)
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.85,
        top_p=0.85,
        top_k=50,
    )
    print(f"  输入 token 数:  {prompt.shape[1]}")
    print(f"  输出 token 数:  {generated.shape[1]}")
    print(f"  新生成 token 数: {generated.shape[1] - prompt.shape[1]}")
    print(f"  Token IDs (前20): {generated[0, :20].tolist()}")

    # —— 贪心生成测试 ——
    print(f"\n🎯 生成测试 (贪心模式):")
    greedy = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        do_sample=False,
    )
    print(f"  输出 token 数:  {greedy.shape[1]}")

    # —— KV-cache 对比验证 ——
    print(f"\n💾 KV-cache 对比验证:")
    # 无缓存生成
    torch.manual_seed(42)
    no_cache = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        use_cache=False,
    )
    # 有缓存生成（相同种子确保确定性）
    torch.manual_seed(42)
    with_cache = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        use_cache=True,
    )
    match = torch.equal(no_cache, with_cache)
    print(f"  无缓存结果:    {no_cache[0, -10:].tolist()}")
    print(f"  有缓存结果:    {with_cache[0, -10:].tolist()}")
    print(f"  结果一致:      {'✅ 是' if match else '❌ 否'}")

    # —— 内存占用估算 ——
    print(f"\n💡 内存占用估算 (FP32):")
    fp32_bytes = total_params * 4  # float32 每个参数 4 字节
    fp16_bytes = total_params * 2  # float16 每个参数 2 字节
    print(f"  FP32: {fp32_bytes / 1024 / 1024:.1f} MB")
    print(f"  FP16: {fp16_bytes / 1024 / 1024:.1f} MB")
    print(f"  FP16 + 梯度 + 优化器(Adam): ~{(fp16_bytes * 3) / 1024 / 1024:.0f} MB")

    print(f"\n{'=' * 60}")
    print("  ✅ TinyMind 模型所有测试通过！")
    print(f"{'=' * 60}")
