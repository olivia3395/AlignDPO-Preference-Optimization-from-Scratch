# DPO Alignment

Fine-tune a causal LLM using **Direct Preference Optimization (DPO)** on
Anthropic HH-RLHF. Also implements **IPO** and **KTO** from scratch for ablation.

---

## Project Structure

| File | Description |
|------|-------------|
| `train_dpo.py` | End-to-end DPO training with QLoRA (4-bit NF4 + LoRA) |
| `dpo_loss.py` | DPO / IPO / KTO loss implemented from scratch |
| `data_utils.py` | HH-RLHF + UltraFeedback loaders and formatters |
| `eval.py` | Reward accuracy, reward margin, GPT-4 win-rate judge |
| `compare_losses.py` | Ablation: DPO vs IPO vs KTO on identical setup |
| `configs/dpo_config.yaml` | All hyperparameters |

---

## Datasets

### Primary — `Anthropic/hh-rlhf` (helpful-base)

Used by default in `train_dpo.py` and `compare_losses.py`.

```python
load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
```

| Property | Value |
|----------|-------|
| Source | Anthropic (human annotators) |
| Size | ~43k preference pairs |
| Format | `{"chosen": "Human: ...\n\nAssistant: ...", "rejected": "..."}` |
| Download size | ~50 MB |
| Access | Public, no approval needed |

The `helpful-base` subset contains human preference judgments on helpfulness only,
giving cleaner signal for a first training run than mixing helpful + harmless.

### Alternative — `HuggingFaceH4/ultrafeedback_binarized`

Higher-quality AI-annotated pairs (GPT-4 scored). Swap in via `data_utils.py`.

```python
# In train_dpo.py, replace:
train_dataset, eval_dataset = load_hh_rlhf(...)
# with:
train_dataset, eval_dataset = load_ultrafeedback(...)
```

| Property | Value |
|----------|-------|
| Source | HuggingFace H4 team (GPT-4 annotations) |
| Size | ~61k preference pairs |
| Format | Conversation list, auto-formatted by `format_ultrafeedback_sample` |
| Download size | ~200 MB |
| Access | Public, no approval needed |

Both datasets download automatically via HuggingFace `datasets` — no local files needed.

---

## Setup

```bash
pip install -r requirements.txt
```

**GPU requirement:** >= 24 GB VRAM (A10G / A100 / RTX 3090).
For < 24 GB: set `per_device_train_batch_size: 1` and `max_length: 768` in the config.

---

## Usage

### 1. Train

```bash
python train_dpo.py --config configs/dpo_config.yaml
```

Trains Mistral-7B-Instruct on HH-RLHF helpful-base with sigmoid DPO loss.
Logs reward accuracy, reward margin, and training loss to W&B.
Saves LoRA adapter to `outputs/dpo_mistral/`.

To switch loss type, edit `loss_type` in the config:

```yaml
loss_type: "ipo"      # or "kto_pair"
```

### 2. Ablation — DPO vs IPO vs KTO

```bash
python compare_losses.py --config configs/dpo_config.yaml
```

Trains three models with identical hyperparameters, different loss functions,
then prints a summary table:

```
Loss          Reward Acc   Reward Margin
--------      ----------   -------------
sigmoid           0.XXXX          X.XXXX
ipo               0.XXXX          X.XXXX
kto_pair          0.XXXX          X.XXXX
```

### 3. Inspect the data

```bash
python data_utils.py
```

Prints a sample prompt / chosen / rejected triple and dataset sizes.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Reward accuracy** | % of eval pairs where implicit chosen reward > rejected |
| **Reward margin** | Mean(r_chosen - r_rejected); larger = better separation |
| **Win rate** | GPT-4-as-judge vs. base model (optional, needs OpenAI key) |

---

## DPO Loss Reference

**Standard DPO (sigmoid):**

$$\mathcal{L} = -\mathbb{E}\left[\log\sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right)\right]$$

**IPO (identity — avoids reward over-optimisation):**

$$\mathcal{L} = \left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2$$

---

## References

- Rafailov et al. (2023) *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- Azar et al. (2023) *A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO)*
- Ethayarajh et al. (2023) *KTO: Model Alignment as Prospect Theoretic Optimization*
