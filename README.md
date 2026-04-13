<div align="center">

# 🧠 DPO Alignment

**Fine-tune causal LLMs with Direct Preference Optimization**

[![Python](https://img.shields.io/badge/Python-3.10+-3572A5?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.x-FFD21E?style=flat-square)](https://huggingface.co/transformers)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

Train Mistral-7B on [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) with QLoRA + DPO.  
Swap loss functions — **DPO · IPO · KTO** — with a single YAML flag.

<br/>

</div>


## Project Structure

```
dpo-alignment/
├── train_dpo.py            # End-to-end DPO training (QLoRA · 4-bit NF4 + LoRA)
├── dpo_loss.py             # DPO / IPO / KTO loss from scratch (educational)
├── data_utils.py           # HH-RLHF & UltraFeedback loaders and formatters
├── eval.py                 # Reward accuracy · reward margin · GPT-4 win-rate
├── compare_losses.py       # Ablation: DPO vs IPO vs KTO on identical setup
└── configs/
    └── dpo_config.yaml     # All hyperparameters in one place
```



## 🗂️ Datasets

Two preference datasets are supported out of the box — both download automatically via HuggingFace `datasets`, no local files needed.

### Primary — `Anthropic/hh-rlhf` *(default)*

```python
load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
```

| Property | Value |
|---|---|
| Source | Anthropic (human annotators) |
| Size | ~43k preference pairs |
| Format | `{"chosen": "Human: …\n\nAssistant: …", "rejected": "…"}` |
| Download | ~50 MB · Public, no approval needed |

The `helpful-base` subset focuses purely on helpfulness, giving a cleaner training signal for a first run than the mixed helpful + harmless split.

### Alternative — `HuggingFaceH4/ultrafeedback_binarized`

Higher-quality AI-annotated pairs (GPT-4 scored). Swap in via `data_utils.py`:

```python
# Replace in train_dpo.py:
train_dataset, eval_dataset = load_hh_rlhf(...)        # ← remove
train_dataset, eval_dataset = load_ultrafeedback(...)   # ← add
```

| Property | Value |
|---|---|
| Source | HuggingFace H4 (GPT-4 annotations) |
| Size | ~61k preference pairs |
| Format | Conversation list, auto-formatted by `format_ultrafeedback_sample` |
| Download | ~200 MB · Public, no approval needed |



## Setup

```bash
pip install -r requirements.txt
```

> **GPU requirement:** ≥ 24 GB VRAM (A10G / A100 / RTX 3090)  
> For < 24 GB — set in `configs/dpo_config.yaml`:
> ```yaml
> per_device_train_batch_size: 1
> max_length: 768
> ```



## 🚀 Usage

### 1 · Train

```bash
python train_dpo.py --config configs/dpo_config.yaml
```

Trains Mistral-7B-Instruct on HH-RLHF `helpful-base` with sigmoid DPO loss.  
Logs reward accuracy, reward margin, and training loss to W&B.  
Saves the LoRA adapter to `outputs/dpo_mistral/`.

To switch loss type, edit one line in the config:

```yaml
loss_type: "ipo"       # sigmoid | ipo | kto_pair
```

### 2 · Ablation — DPO vs IPO vs KTO

```bash
python compare_losses.py --config configs/dpo_config.yaml
```

Trains three models with identical hyperparameters but different loss functions, logs everything to W&B, and prints a summary table:

```
Loss          Reward Acc    Reward Margin
──────────    ──────────    ─────────────
sigmoid           0.XXXX           X.XXXX
ipo               0.XXXX           X.XXXX
kto_pair          0.XXXX           X.XXXX
```

### 3 · Inspect the data

```bash
python data_utils.py
```

Prints a sample prompt / chosen / rejected triple and reports dataset sizes.



## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Reward accuracy** | % of eval pairs where implicit chosen reward > rejected |
| **Reward margin** | Mean(r_chosen − r_rejected) — larger = better separation |
| **Win rate** | GPT-4-as-judge pairwise vs. base model *(optional — needs OpenAI key)* |



## Loss Functions

### DPO — Direct Preference Optimization *(sigmoid)*

Maximises the log-probability gap between chosen and rejected completions, regularised against a frozen reference policy.



### IPO — Identity Preference Optimization

Replaces the sigmoid with a squared loss, directly targeting the preference gap and avoiding reward over-optimisation.

### KTO — Kahneman-Tversky Optimization

Reframes alignment as prospect-theoretic utility maximisation. Unlike DPO/IPO, operates on **binary desirability labels** — no pairwise preferences required. See `dpo_loss.py` for the full derivation.



## 📚 References

<details>
<summary><b>Rafailov et al. (2023)</b> &nbsp;·&nbsp; Direct Preference Optimization</summary>
<br/>

*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*  
Shows that the RLHF objective can be re-expressed as a simple binary cross-entropy loss over preference pairs, eliminating the need for explicit reward modelling.

</details>

<details>
<summary><b>Azar et al. (2023)</b> &nbsp;·&nbsp; IPO</summary>
<br/>

*A General Theoretical Paradigm to Understand Learning from Human Feedback*  
Introduces the identity-link loss variant that targets the preference gap directly and is more robust to reward over-optimisation.

</details>

<details>
<summary><b>Ethayarajh et al. (2023)</b> &nbsp;·&nbsp; KTO</summary>
<br/>

*KTO: Model Alignment as Prospect Theoretic Optimization*  
Applies Kahneman–Tversky utility theory to alignment; works without pairwise comparisons, only requiring per-response desirability labels.

</details>



<div align="center">


</div>
