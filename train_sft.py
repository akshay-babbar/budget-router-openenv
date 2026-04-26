#!/usr/bin/env python3
# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45.0",
#   "trl>=0.25.0",
#   "peft>=0.13.0",
#   "datasets>=2.20.0",
#   "accelerate>=0.34.0",
#   "huggingface_hub>=0.24.0",
# ]
# ///
"""Train a LoRA SFT Budget Router model on HF Jobs and push merged weights."""

from __future__ import annotations

import argparse
import os


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_DATASET_REPO = "akshay4/budget-router-sft-data"
DEFAULT_OUTPUT_REPO = "akshay4/budget-router-sft-qwen1.5b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Budget Router SFT model.")
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--dataset-repo", default=os.getenv("DATASET_REPO", DEFAULT_DATASET_REPO))
    parser.add_argument("--output-repo", default=os.getenv("OUTPUT_REPO", DEFAULT_OUTPUT_REPO))
    parser.add_argument("--num-epochs", type=float, default=float(os.getenv("NUM_EPOCHS", "3")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "2e-4")))
    parser.add_argument("--lora-r", type=int, default=int(os.getenv("LORA_R", "16")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.getenv("LORA_ALPHA", "32")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_SEQ_LENGTH", "4096")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("PER_DEVICE_BATCH_SIZE", "2")))
    parser.add_argument("--grad-accum", type=int, default=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN must be set as a secret in the HF Job.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    device_supports_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if device_supports_bf16 else torch.float16

    print(f"[train-sft] loading dataset={args.dataset_repo}", flush=True)
    dataset = load_dataset(args.dataset_repo, split="train", token=token)
    print(f"[train-sft] rows={len(dataset)}", flush=True)

    print(f"[train-sft] loading base_model={args.base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir="./sft_output",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=device_supports_bf16,
        fp16=not device_supports_bf16,
        optim="adamw_torch",
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        max_length=args.max_length,
        packing=False,
        assistant_only_loss=True,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainable = getattr(trainer.model, "print_trainable_parameters", None)
    if callable(trainable):
        trainable()

    print("[train-sft] starting training", flush=True)
    train_result = trainer.train()
    print(f"[train-sft] train_metrics={train_result.metrics}", flush=True)

    final_loss = train_result.metrics.get("train_loss")
    if final_loss is not None and float(final_loss) > 0.5:
        print("[train-sft] WARNING: train_loss > 0.5; inspect data and consider more epochs.", flush=True)

    print("[train-sft] merging LoRA and pushing model", flush=True)
    merged = trainer.model.merge_and_unload() if hasattr(trainer.model, "merge_and_unload") else trainer.model
    merged.push_to_hub(args.output_repo, token=token)
    tokenizer.push_to_hub(args.output_repo, token=token)
    print(f"[train-sft] Model pushed to {args.output_repo}. Training complete.", flush=True)


if __name__ == "__main__":
    main()
