<div align="center">

# 🧠 DPO Alignment

**Fine-tune causal LLMs with Direct Preference Optimization — plus manual IPO & KTO for ablation**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![VRAM](https://img.shields.io/badge/VRAM-≥%2024%20GB-8b5cf6?style=flat-square&logo=nvidia&logoColor=white)]()
[![W&B](https://img.shields.io/badge/Logging-Weights%20%26%20Biases-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

<br/>

> Train preference-aligned language models using DPO on the **Anthropic HH-RLHF** dataset,  
> with educational from-scratch implementations of **IPO** and **KTO** for direct comparison.

</div>

---

## ✨ Highlights

- 🔧 **QLoRA (4-bit NF4 + LoRA)** — memory-efficient fine-tuning on consumer GPUs
- 📐 **Three loss functions from scratch** — DPO, IPO, and KTO, fully hand-rolled
- 📊 **Automated ablation runner** — identical setups, one command, W&B comparison table
- 🏆 **GPT-4-as-judge eval** — optional pairwise win-rate scoring vs. base model
- 🗂️ **Multi-dataset support** — HH-RLHF and UltraFeedback loaders included

---

## 📁 Project Structure

```
dpo-alignment/
├── train_dpo.py          # End-to-end DPO training pipeline
├── dpo_loss.py           # DPO / IPO / KTO loss implementations (from scratch)
├── data_utils.py         # HH-RLHF & UltraFeedback loaders and formatters
├── eval.py               # Reward accuracy, margin & GPT-4 win-rate judge
├── compare_losses.py     # Ablation: DPO vs IPO vs KTO on identical setup
├── configs/
│   └── dpo_config.yaml   # Hyperparameter configuration
└── requirements.txt
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

> **GPU Requirements**
> | Configuration | Minimum VRAM |
> |---|---|
> | Default (recommended) | ≥ 24 GB &nbsp;(A10G / A100 / RTX 3090) |
> | Reduced (`batch_size=1`, `max_length=768`) | < 24 GB |

---

## 🚀 Training

#### Standard DPO (Mistral-7B + HH-RLHF + sigmoid loss)

```bash
python train_dpo.py --config configs/dpo_config.yaml
```

#### Switch loss type on the fly

Edit `loss_type` inside `configs/dpo_config.yaml`:

```yaml
# options: "sigmoid" | "ipo" | "kto"
loss_type: sigmoid
```

#### Run the full ablation (DPO vs IPO vs KTO)

```bash
python compare_losses.py --config configs/dpo_config.yaml
```

Trains three identical models with different loss functions, logs all metrics to W&B, and prints a final summary table.

---

## 📐 Loss Functions

<details>
<summary><strong>DPO — Standard Sigmoid Loss</strong></summary>

<br/>

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\!\left(\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right)\right]$$

Maximises the margin between chosen and rejected log-ratio differences via a binary cross-entropy objective.

</details>

<details>
<summary><strong>IPO — Identity / Squared Loss</strong></summary>

<br/>

$$\mathcal{L}_{\text{IPO}} = \left(\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\beta}\right)^{\!2}$$

A theoretically grounded alternative that avoids over-optimisation by targeting a specific preference gap rather than maximising it unboundedly.

</details>

<details>
<summary><strong>KTO — Prospect-Theoretic Loss</strong></summary>

<br/>

KTO aligns models using individual scalar reward signals (good / bad) rather than contrastive pairs, drawing on Kahneman-Tversky prospect theory to model human loss aversion.

</details>

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| **Reward Accuracy** | % of eval pairs where chosen reward > rejected reward |
| **Reward Margin** | Mean( r_chosen − r_rejected ) across the eval set |
| **Win Rate** | GPT-4-as-judge pairwise comparison vs. base model *(optional)* |

Run evaluation on a trained checkpoint:

```bash
python eval.py --model_path ./outputs/dpo-checkpoint --config configs/dpo_config.yaml
```

---

## 📚 References

| Paper | Authors | Year |
|---|---|---|
| [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | Rafailov et al. | 2023 |
| [A General Theoretical Paradigm to Understand Learning from Human Feedback (IPO)](https://arxiv.org/abs/2310.12036) | Azar et al. | 2023 |
| [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) | Ethayarajh et al. | 2023 |

---

<div align="center">


</div>
