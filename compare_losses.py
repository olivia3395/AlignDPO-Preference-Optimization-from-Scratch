"""
compare_losses.py
-----------------
Ablation: train DPO / IPO / KTO on the same base model + data,
then compare reward accuracy, margin, and win rate.

Run:
    python compare_losses.py --config configs/dpo_config.yaml
"""

import torch
import wandb
import argparse
import yaml
import copy

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from data_utils import load_hh_rlhf, format_dpo_sample
from eval import compute_implicit_rewards


LOSS_TYPES = ["sigmoid", "ipo", "kto_pair"]


def build_trainer(base_model_name, loss_type, train_ds, eval_ds, tokenizer,
                  output_dir, cfg):
    """Build a fresh DPOTrainer for a given loss type."""

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    dpo_args = DPOConfig(
        beta=cfg["beta"],
        loss_type=loss_type,
        max_prompt_length=cfg["max_prompt_length"],
        max_length=cfg["max_length"],
        output_dir=output_dir,
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        fp16=True,
        logging_steps=cfg["logging_steps"],
        seed=cfg["seed"],
        report_to="wandb",
    )

    return DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw_train, raw_eval = load_hh_rlhf(split_ratio=0.02)
    train_ds = raw_train.map(format_dpo_sample, remove_columns=raw_train.column_names)
    eval_ds  = raw_eval.map(format_dpo_sample,  remove_columns=raw_eval.column_names)

    all_metrics = {}

    for loss_type in LOSS_TYPES:
        print(f"\n{'='*60}")
        print(f"Training with loss_type = {loss_type}")
        print(f"{'='*60}")

        out_dir = f"./outputs/{loss_type}"

        wandb.init(
            project=cfg["wandb_project"],
            name=f"{cfg['run_name']}-{loss_type}",
            config={**cfg, "loss_type": loss_type},
            reinit=True,
        )

        trainer = build_trainer(
            cfg["model_name"], loss_type,
            train_ds, eval_ds, tokenizer, out_dir, cfg
        )
        trainer.train()
        trainer.save_model(out_dir)

        # Evaluate
        metrics = compute_implicit_rewards(
            trainer.model,
            trainer.ref_model,
            tokenizer,
            eval_ds.select(range(200)),
            beta=cfg["beta"],
        )
        metrics["loss_type"] = loss_type
        all_metrics[loss_type] = metrics
        wandb.log(metrics)
        wandb.finish()

        del trainer
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n=== ABLATION RESULTS ===")
    header = f"{'Loss':<12} {'Reward Acc':>12} {'Reward Margin':>14}"
    print(header)
    print("-" * len(header))
    for lt, m in all_metrics.items():
        print(f"{lt:<12} {m['reward_accuracy']:>12.4f} {m['reward_margin']:>14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()
    main(args.config)
