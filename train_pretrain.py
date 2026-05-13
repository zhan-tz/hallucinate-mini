"""
TinyMind 预训练脚本
====================
从零开始训练一个小型GPT模型，使用纯文本JSONL数据进行下一个token预测。

用法:
    cd tinymind && python train_pretrain.py
    cd tinymind && python train_pretrain.py --epochs 3 --batch_size 16 --lr 1e-4

特性:
    - 自动检测设备 (MPS / CUDA / CPU)
    - 余弦学习率调度 (warmup + cosine decay)
    - 梯度累积支持有效更大batch size
    - 定期保存检查点
    - 训练结束后快速测试生成
"""

import os
import sys
import json
import math
import time
import argparse
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# 确保可以从 tinymind/ 目录导入本地 model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMindConfig, TinyMindForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# 学习率调度器: 带warmup的余弦衰减
# ============================================================================
def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """
    余弦学习率调度: warmup阶段线性增长，之后余弦衰减。
    lr * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
    范围约在 [0.1*lr, lr] 之间。
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


# ============================================================================
# 预训练数据集: 读取JSONL格式 {"text": "..."}
# ============================================================================
class PretrainDataset(Dataset):
    """预训练数据集，每行为 {"text": "..."} 的JSONL文件。"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_id = tokenizer.bos_token_id  # <|im_start|>
        self.eos_id = tokenizer.eos_token_id  # <|im_end|>
        self.pad_id = tokenizer.pad_token_id   # <|endoftext|>

        # 逐行读取JSONL，避免依赖datasets库
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        print(f"已加载 {len(self.samples)} 条预训练样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample["text"]

        # 分词，预留 BOS/EOS token 的位置
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
        )["input_ids"]

        # 拼接: [BOS] + tokens + [EOS]
        token_ids = [self.bos_id] + tokens + [self.eos_id]

        # Padding 到固定长度
        if len(token_ids) < self.max_length:
            pad_len = self.max_length - len(token_ids)
            token_ids = token_ids + [self.pad_id] * pad_len
        else:
            token_ids = token_ids[: self.max_length]

        input_ids = torch.tensor(token_ids, dtype=torch.long)

        # labels: 与 input_ids 相同，但 padding 位置设为 -100（忽略）
        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100

        return input_ids, labels


# ============================================================================
# 训练一个epoch
# ============================================================================
def train_epoch(epoch, loader, total_steps, start_step=0):
    """训练一个epoch，返回该epoch的平均loss。"""
    model.train()
    start_time = time.time()
    total_loss = 0.0
    step_count = 0
    accumulated_loss = 0.0

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        # 计算当前学习率
        global_step = epoch * total_steps + step
        lr = get_lr(global_step, args.epochs * total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 前向传播
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss = loss / args.accumulation_steps

        # 反向传播
        loss.backward()

        accumulated_loss += loss.item() * args.accumulation_steps
        step_count += 1

        # 梯度累积: 每 accumulation_steps 步更新一次参数
        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0 or step == total_steps:
            spend_time = time.time() - start_time
            avg_loss = accumulated_loss / step_count
            current_lr = optimizer.param_groups[-1]["lr"]

            # 估算剩余时间
            steps_done = step - start_step
            eta_sec = spend_time / max(steps_done, 1) * (total_steps - step)
            eta_min = eta_sec // 60
            eta_sec %= 60

            print(
                f"[Epoch {epoch + 1}/{args.epochs}] "
                f"Step {step}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"ETA: {eta_min:.0f}m {eta_sec:.0f}s"
            )

            total_loss += accumulated_loss
            accumulated_loss = 0.0
            step_count = 0

        # 定期保存检查点
        if (step % args.save_interval == 0 or step == total_steps) and step > 0:
            save_checkpoint(epoch, step)

        del input_ids, labels, outputs, loss

    # 处理最后不足 accumulation_steps 的残留梯度
    if step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_steps, 1)


# ============================================================================
# 保存模型检查点
# ============================================================================
def save_checkpoint(epoch, step):
    """保存模型权重到 ./out/ 目录。"""
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"pretrain_{config.d_model}.pth")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, save_path)
    print(f"  检查点已保存: {save_path}")


# ============================================================================
# 快速测试生成
# ============================================================================
@torch.no_grad()
def test_generate(prompt: str = "今天天气真不错", max_new_tokens: int = 30):
    """训练结束后用当前模型生成一段文字作为快速测试。"""
    print("\n" + "=" * 60)
    print("训练结束，快速生成测试 (预训练基座模型不按对话格式)")
    print("=" * 60)

    model.eval()
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    token_ids = [tokenizer.bos_token_id] + token_ids
    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)

    try:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.85,
            top_p=0.85,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
        generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {generated_text}")
    except Exception as e:
        print(f"生成测试失败: {e}")


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyMind 预训练")
    # 模型配置
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度(tokens)")
    # 训练配置
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="每批次样本数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数(有效batch=batch_size*accumulation)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔(步数)")
    parser.add_argument("--save_interval", type=int, default=500, help="检查点保存间隔(步数)")
    # 路径配置
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径(JSONL)")
    parser.add_argument("--tokenizer_path", type=str, default="./", help="Tokenizer目录路径")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    # 其他
    parser.add_argument("--device", type=str, default=None, help="训练设备 (auto/cpu/cuda/mps)")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数 (MPS建议设为0)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # ==============================
    # 1. 设备检测
    # ==============================
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"使用设备: {device}")
    if device == "mps":
        print("  注意: MPS 不支持混合精度训练, 使用 float32")

    # ==============================
    # 2. 随机种子
    # ==============================
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ==============================
    # 3. 加载 Tokenizer
    # ==============================
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Tokenizer词表大小: {tokenizer.vocab_size}")
    print(f"  BOS token: '{tokenizer.bos_token}' (id={tokenizer.bos_token_id})")
    print(f"  EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"  PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

    # ==============================
    # 4. 创建模型
    # ==============================
    config = TinyMindConfig(
        d_model=args.hidden_size,
        n_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len * 4,
    )
    model = TinyMindForCausalLM(config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")

    # ==============================
    # 5. 加载数据
    # ==============================
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        sys.exit(1)

    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    steps_per_epoch = len(train_loader)
    effective_batch = args.batch_size * args.accumulation_steps
    print(f"\n训练配置:")
    print(f"  样本总数: {len(train_ds)}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  accumulation_steps: {args.accumulation_steps}")
    print(f"  有效batch大小: {effective_batch}")
    print(f"  每epoch步数: {steps_per_epoch}")
    print(f"  总训练步数: {steps_per_epoch * args.epochs}")
    print(f"  max_seq_len: {args.max_seq_len}")

    # ==============================
    # 6. 优化器
    # ==============================
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ==============================
    # 7. 训练循环
    # ==============================
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print("开始预训练")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        epoch_loss = train_epoch(epoch, train_loader, steps_per_epoch)
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs} 完成, 平均Loss: {epoch_loss:.4f}")

    # ==============================
    # 8. 保存最终模型 & 快速测试
    # ==============================
    save_checkpoint(args.epochs - 1, steps_per_epoch)
    test_generate()

    print("\n预训练完成!")
