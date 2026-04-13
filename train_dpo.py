"""
train_dpo.py
------------
DPO fine-tuning with QLoRA on Anthropic HH-RLHF (helpful-base subset).
Base model: mistralai/Mistral-7B-Instruct-v0.2

Usage:
    python train_dpo.py --config configs/dpo_config.yaml

Data flow:
    Anthropic/hh-rlhf (helpful-base)
        → format_dpo_sample()
        → {"prompt", "chosen", "rejected"}
        → DPOTrainer (TRL)
"""

import os
import yaml
import argparse
import torch
import wandb

from dataclasses import dataclass, field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from data_utils import load_hh_rlhf
from eval import run_eval


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DPOTrainConfig:
    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj",
        ]
    )

    # DPO
    beta: float = 0.1
    max_prompt_length: int = 512
    max_length: int = 1024
    loss_type: str = "sigmoid"   # "sigmoid" | "ipo" | "kto_pair"

    # Training
    output_dir: str = "./outputs/dpo_mistral"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    fp16: bool = True
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    seed: int = 42

    # W&B
    wandb_project: str = "dpo-alignment"
    run_name: str = "mistral-dpo-hh"


def load_config(path: str) -> DPOTrainConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DPOTrainConfig(**raw)


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model_and_tokenizer(cfg: DPOTrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO requires left-padding

    bnb_cfg = None
    if cfg.use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: DPOTrainConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading Anthropic/hh-rlhf (helpful-base)...")
    train_dataset, eval_dataset = load_hh_rlhf(split_ratio=0.02, subset="helpful-base")
    print(f"Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")
    print("Sample:", train_dataset[0])

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = build_model_and_tokenizer(cfg)

    # ── DPO Trainer ──────────────────────────────────────────────────────────
    # DPOConfig inherits from TrainingArguments (TRL >= 0.8).
    # Use eval_strategy (new name) with fallback for older transformers.
    dpo_args = DPOConfig(
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_prompt_length=cfg.max_prompt_length,
        max_length=cfg.max_length,
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",          # replaces evaluation_strategy in newer versions
        save_total_limit=2,
        seed=cfg.seed,
        report_to="wandb",
        remove_unused_columns=False,    # required for DPO
    )

    # TRL >= 0.9 uses processing_class; fall back to tokenizer for older versions
    try:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"Model saved to {cfg.output_dir}")

    # ── Eval ─────────────────────────────────────────────────────────────────
    metrics = run_eval(
        model_path=cfg.output_dir,
        base_model_name=cfg.model_name,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        beta=cfg.beta,
        n_samples=200,
    )
    wandb.log(metrics)
    print("Final metrics:", metrics)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)
