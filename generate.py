"""
TinyMind 推理/生成脚本
========================
加载训练好的模型进行文本生成，支持交互对话和自动测试两种模式。

用法:
    cd tinymind && python generate.py --weight pretrain      # 预训练模型自动测试
    cd tinymind && python generate.py --weight full_sft       # SFT模型交互对话
    cd tinymind && python generate.py --weight full_sft --mode auto --temperature 0.9

特性:
    - 自动检测设备 (MPS / CUDA / CPU)
    - 交互式对话模式 (SFT模型) / 文本续写模式 (预训练模型)
    - temperature / top-p / top-k 采样参数可调
    - 显示生成速度 (tokens/sec)
    - 内置中文测试提示词集
"""

import os
import sys
import time
import argparse
import warnings

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import TinyMindConfig, TinyMindForCausalLM

warnings.filterwarnings("ignore")

DEFAULT_TEST_PROMPTS = [
    "请介绍一下你自己。",
    "为什么天空是蓝色的？",
    "请用Python写一个斐波那契数列的计算函数。",
    "请解释什么是光合作用。",
    "推荐一些中国的美食。",
    "什么是机器学习？",
    "请用简短的语言总结一下《西游记》的故事。",
    "写一首关于秋天的五言诗。",
]


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(weight: str, tokenizer_path: str, save_dir: str, hidden_size: int = 256, num_hidden_layers: int = 4):
    """加载tokenizer、模型配置和训练好的权重。"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    config = TinyMindConfig(
        d_model=hidden_size,
        n_layers=num_hidden_layers,
        vocab_size=tokenizer.vocab_size,
    )
    model = TinyMindForCausalLM(config)

    weight_path = os.path.join(save_dir, f"{weight}_{config.d_model}.pth")
    if not os.path.exists(weight_path):
        print(f"错误: 模型权重不存在: {weight_path}")
        print(f"  请先在 ./out/ 目录下放入对应的 .pth 文件")
        sys.exit(1)

    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {total_params:.2f}M")
    print(f"已加载权重: {weight_path}")

    return model, tokenizer, config


@torch.no_grad()
def generate_text(
    model, tokenizer, prompt: str, device: str,
    temperature: float = 0.85, top_p: float = 0.85, top_k: int = 50,
    max_new_tokens: int = 256, is_sft: bool = True,
) -> tuple:
    """
    文本生成核心函数。
    返回 (生成的文本, 耗时秒数, 生成token数)。
    """
    if is_sft:
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        full_prompt = f"{bos}user\n{prompt}{eos}\n{bos}assistant\n"
    else:
        full_prompt = prompt

    input_ids = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    prompt_len = input_ids.shape[1]

    start_time = time.time()
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=(temperature > 0),
        use_cache=True,
    )
    elapsed = time.time() - start_time

    generated_tokens = output_ids.shape[1] - prompt_len
    full_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    if is_sft:
        response = tokenizer.decode(
            output_ids[0, prompt_len:].tolist(), skip_special_tokens=True
        )
    else:
        response = full_text

    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0
    return response, elapsed, generated_tokens, tokens_per_sec


def auto_test(model, tokenizer, config, device: str, args):
    """自动测试模式: 遍历内置提示词逐个生成。"""
    is_sft = args.weight.startswith("full_sft") or args.weight.startswith("sft")
    print(f"\n{'=' * 60}")
    print(f"自动测试模式 (模型: {args.weight})")
    print(f"temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print(f"{'=' * 60}")

    for i, prompt in enumerate(DEFAULT_TEST_PROMPTS):
        print(f"\n--- 测试 {i + 1}/{len(DEFAULT_TEST_PROMPTS)} ---")
        print(f"User: {prompt}")

        response, elapsed, n_tokens, tps = generate_text(
            model, tokenizer, prompt, device,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, is_sft=is_sft,
        )
        print(f"Model: {response}")
        print(f"[{n_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tokens/s]")


def interactive_chat(model, tokenizer, config, device: str, args):
    """交互式对话模式: 循环读取用户输入并生成回复。"""
    is_sft = args.weight.startswith("full_sft") or args.weight.startswith("sft")
    print(f"\n{'=' * 60}")
    print(f"交互对话模式 (模型: {args.weight})")
    print(f"输入 'quit' 或 'exit' 退出, 'clear' 清屏")
    print(f"{'=' * 60}")

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见!")
            break
        if user_input.lower() == "clear":
            os.system("clear" if os.name != "nt" else "cls")
            continue

        response, elapsed, n_tokens, tps = generate_text(
            model, tokenizer, user_input, device,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            max_new_tokens=args.max_new_tokens, is_sft=is_sft,
        )
        print(f"Model: {response}")
        print(f"[{n_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tokens/s]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyMind 推理生成")
    parser.add_argument("--weight", type=str, default="full_sft", help="权重名 (pretrain / full_sft)")
    parser.add_argument("--tokenizer_path", type=str, default="./", help="Tokenizer目录路径")
    parser.add_argument("--save_dir", type=str, default="./out", help="模型权重存放目录")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "chat"], help="运行模式: auto=自动测试, chat=交互对话")
    parser.add_argument("--temperature", type=float, default=0.85, help="采样温度 (0=贪心)")
    parser.add_argument("--top_p", type=float, default=0.85, help="nucleus采样概率阈值")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样候选数")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成token数")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层维度 (需与训练时一致)")
    parser.add_argument("--num_hidden_layers", type=int, default=4, help="Transformer层数 (需与训练时一致)")
    parser.add_argument("--device", type=str, default=None, help="设备 (auto/cpu/cuda/mps)")
    parser.add_argument("--prompt", type=str, default=None, help="自定义单次提示词 (仅在auto模式下生效)")

    args = parser.parse_args()

    device = args.device or detect_device()
    print(f"使用设备: {device}")

    model, tokenizer, config = load_model(
        args.weight, args.tokenizer_path, args.save_dir,
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
    )
    model = model.to(device)

    if args.mode == "chat":
        interactive_chat(model, tokenizer, config, device, args)
    else:
        if args.prompt:
            is_sft = args.weight.startswith("full_sft") or args.weight.startswith("sft")
            print(f"\nUser: {args.prompt}")
            response, elapsed, n_tokens, tps = generate_text(
                model, tokenizer, args.prompt, device,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                max_new_tokens=args.max_new_tokens, is_sft=is_sft,
            )
            print(f"Model: {response}")
            print(f"[{n_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tokens/s]")
        else:
            auto_test(model, tokenizer, config, device, args)
