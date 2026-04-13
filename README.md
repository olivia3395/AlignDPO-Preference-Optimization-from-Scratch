# DPO Alignment

Fine-tune a causal LLM using **Direct Preference Optimization (DPO)**
on the Anthropic HH-RLHF dataset.  
Also includes manual implementations of **IPO** and **KTO** for ablation.

---

## What This Project Covers

| Component | Description |
|-----------|-------------|
| `train_dpo.py` | End-to-end DPO training with QLoRA (4-bit NF4 + LoRA) |
| `dpo_loss.py` | Manual DPO / IPO / KTO loss from scratch (educational) |
| `data_utils.py` | HH-RLHF + UltraFeedback loaders & formatters |
| `eval.py` | Reward accuracy, reward margin, GPT-4 win-rate judge |
| `compare_losses.py` | Ablation: DPO vs IPO vs KTO on identical setup |
| `configs/` | YAML hyperparameter configs |

---

## Setup

```bash
pip install -r requirements.txt
```

GPU requirement: ≥ 24 GB VRAM (A10G / A100 / RTX 3090)  
For < 24 GB: reduce `per_device_train_batch_size` to 1 and `max_length` to 768.

---

## Training

```bash
# DPO with default config (Mistral-7B, HH-RLHF, sigmoid loss)
python train_dpo.py --config configs/dpo_config.yaml

# Swap loss type on the fly
python train_dpo.py --config configs/dpo_config.yaml  # edit loss_type in yaml
```

## Ablation: DPO vs IPO vs KTO

```bash
python compare_losses.py --config configs/dpo_config.yaml
```

Trains three identical models with different loss functions, logs all metrics
to W&B, and prints a summary table.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Reward accuracy** | % of eval pairs where chosen reward > rejected |
| **Reward margin** | Mean(r_chosen − r_rejected) |
| **Win rate** | GPT-4-as-judge pairwise vs. base model (optional) |

---

## DPO Loss (Quick Reference)

**Standard DPO (sigmoid):**
$$\mathcal{L} = -\mathbb{E}\left[\log\sigma\!\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right)\right]$$

**IPO (identity):**
$$\mathcal{L} = \left(\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2$$

---

## References

- Rafailov et al. (2023) *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
- Azar et al. (2023) *A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO)*
- Ethayarajh et al. (2023) *KTO: Model Alignment as Prospect Theoretic Optimization*
