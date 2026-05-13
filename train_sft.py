"""
TinyMind 监督微调脚本
======================
在预训练模型基础上进行对话微调，让模型学会问答交互。

用法:
    cd tinymind && python train_sft.py
    cd tinymind && python train_sft.py --from_weight none --epochs 5

特性:
    - 加载预训练权重，在对话数据上继续训练
    - 自动构建聊天模板 (<|im_start|>role\ncontent<|im_end|>)
    - 仅对assistant回复计算损失，user/system部分被忽略
    - 较低学习率防止灾难性遗忘
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMindConfig, TinyMindForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_lr(current_step: int, total_steps: int, lr: float) -> float:
    """余弦学习率调度 (与预训练相同)"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


class SFTDataset(Dataset):
    """
    SFT数据集: 读取JSONL格式的对话数据。
    格式: {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id

        with open(data_path, "r", encoding="utf-8") as f:
            raw = [json.loads(line.strip()) for line in f if line.strip()]
        self.samples = raw
        print(f"已加载 {len(self.samples)} 条SFT样本")

        self.bos_token = tokenizer.bos_token       # <|im_start|>
        self.eos_token = tokenizer.eos_token       # <|im_end|>

        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        )["input_ids"]
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        )["input_ids"]

    def __len__(self):
        return len(self.samples)

    def build_chat_prompt(self, conversations) -> str:
        """将对话列表拼接为聊天模板文本。"""
        prompt = ""
        for msg in conversations:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{self.bos_token}{role}\n{content}{self.eos_token}\n"
        return prompt

    def generate_labels(self, input_ids: list) -> list:
        """
        生成labels: 仅assistant回复部分参与损失计算。
        扫描input_ids找到<|im_start|>assistant\n的位置，
        标记其后的内容为可学习（直到<|im_end|>\n）。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # assistant内容起始位置
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample["conversations"]

        prompt = self.build_chat_prompt(conversations)
        token_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        token_ids = token_ids[: self.max_length]
        pad_len = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * pad_len

        labels = self.generate_labels(token_ids)

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def train_epoch(epoch, loader, total_steps, start_step=0):
    model.train()
    start_time = time.time()
    total_loss = 0.0
    step_count = 0
    accumulated_loss = 0.0

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        global_step = epoch * total_steps + step
        lr = get_lr(global_step, args.epochs * total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        loss = loss / args.accumulation_steps

        loss.backward()

        accumulated_loss += loss.item() * args.accumulation_steps
        step_count += 1

        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == total_steps:
            spend_time = time.time() - start_time
            avg_loss = accumulated_loss / step_count
            current_lr = optimizer.param_groups[-1]["lr"]

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

        if (step % args.save_interval == 0 or step == total_steps) and step > 0:
            save_checkpoint(epoch, step)

        del input_ids, labels, outputs, loss

    if step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_steps, 1)


def save_checkpoint(epoch, step):
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"full_sft_{config.d_model}.pth")
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, save_path)
    print(f"  检查点已保存: {save_path}")


@torch.no_grad()
def test_chat(prompt: str = "你好，请介绍一下自己"):
    """SFT训练结束后快速测试对话能力。"""
    print("\n" + "=" * 60)
    print("SFT训练结束，对话测试")
    print("=" * 60)

    model.eval()
    chat_text = (
        f"{tokenizer.bos_token}user\n{prompt}{tokenizer.eos_token}\n"
        f"{tokenizer.bos_token}assistant\n"
    )
    input_ids = tokenizer(chat_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    try:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=60,
            temperature=0.85,
            top_p=0.85,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
        response = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"User: {prompt}")
        print(f"Model: {response}")
    except Exception as e:
        print(f"对话测试失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyMind SFT 微调")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="Transformer层数")
    parser.add_argument("--max_seq_len", type=int, default=256, help="最大序列长度(tokens)")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="每批次样本数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率 (SFT用较低lr)")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="检查点保存间隔")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="SFT数据路径")
    parser.add_argument("--tokenizer_path", type=str, default="./", help="Tokenizer目录路径")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型保存目录")
    parser.add_argument("--from_weight", type=str, default="pretrain", help="基于哪个权重训练 (none=从头开始)")
    parser.add_argument("--device", type=str, default=None, help="训练设备 (auto/cpu/cuda/mps)")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

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

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"Tokenizer词表大小: {tokenizer.vocab_size}")

    config = TinyMindConfig(
        d_model=args.hidden_size,
        n_layers=args.num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len * 4,
    )
    model = TinyMindForCausalLM(config)

    if args.from_weight != "none":
        weight_path = os.path.join(args.save_dir, f"{args.from_weight}_{config.d_model}.pth")
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"已加载预训练权重: {weight_path}")
        else:
            print(f"警告: 预训练权重不存在 {weight_path}, 从头开始训练")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")

    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        sys.exit(1)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("开始SFT训练")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        epoch_loss = train_epoch(epoch, train_loader, steps_per_epoch)
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs} 完成, 平均Loss: {epoch_loss:.4f}")

    save_checkpoint(args.epochs - 1, steps_per_epoch)
    test_chat()

    print("\nSFT训练完成!")
