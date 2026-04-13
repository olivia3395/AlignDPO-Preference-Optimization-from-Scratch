"""
compare_losses.py
-----------------
Ablation: train DPO / IPO / KTO on identical setup, compare metrics.

Run:
    python compare_losses.py --config configs/dpo_config.yaml
"""

import torch
import wandb
import argparse
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from data_utils import load_hh_rlhf
from eval import compute_implicit_rewards


LOSS_TYPES = ["sigmoid", "ipo", "kto_pair"]


def build_trainer(base_model_name, loss_type, train_ds, eval_ds, tokenizer,
                  output_dir, cfg):
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
        eval_strategy="no",       # skip mid-training eval in ablation for speed
        seed=cfg["seed"],
        report_to="wandb",
        remove_unused_columns=False,
    )

    try:
        return DPOTrainer(
            model=model, ref_model=None, args=dpo_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            processing_class=tokenizer,
        )
    except TypeError:
        return DPOTrainer(
            model=model, ref_model=None, args=dpo_args,
            train_dataset=train_ds, eval_dataset=eval_ds,
            tokenizer=tokenizer,
        )


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading dataset...")
    train_ds, eval_ds = load_hh_rlhf(split_ratio=0.02, subset="helpful-base")
    print(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")

    # Use a smaller eval subset for the ablation to save time
    eval_subset = eval_ds.select(range(min(200, len(eval_ds))))

    all_metrics = {}

    for loss_type in LOSS_TYPES:
        print(f"\n{'='*60}\nTraining: loss_type = {loss_type}\n{'='*60}")
        out_dir = f"./outputs/{loss_type}"

        wandb.init(
            project=cfg["wandb_project"],
            name=f"{cfg['run_name']}-{loss_type}",
            config={**cfg, "loss_type": loss_type},
            reinit=True,
        )

        trainer = build_trainer(
            cfg["model_name"], loss_type,
            train_ds, eval_ds, tokenizer, out_dir, cfg,
        )
        trainer.train()
        trainer.save_model(out_dir)

        # Load a fresh reference model for eval (don't rely on trainer internals)
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], device_map="auto", torch_dtype=torch.float16
        )

        metrics = compute_implicit_rewards(
            trainer.model, ref_model, tokenizer,
            eval_subset, beta=cfg["beta"],
        )
        metrics["loss_type"] = loss_type
        all_metrics[loss_type] = metrics
        wandb.log(metrics)
        wandb.finish()

        del trainer, ref_model
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n=== ABLATION RESULTS ===")
    header = f"{'Loss':<12} {'Reward Acc':>12} {'Reward Margin':>15}"
    print(header)
    print("-" * len(header))
    for lt, m in all_metrics.items():
        print(f"{lt:<12} {m['reward_accuracy']:>12.4f} {m['reward_margin']:>15.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_config.yaml")
    args = parser.parse_args()
    main(args.config)
