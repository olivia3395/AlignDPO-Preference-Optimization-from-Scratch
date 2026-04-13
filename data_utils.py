"""
data_utils.py
-------------
Load and format preference datasets into the triplet format DPOTrainer expects:
    {"prompt": str, "chosen": str, "rejected": str}

Datasets
--------
- Anthropic/hh-rlhf          (161k human preference pairs, helpfulness+harmlessness)
- HuggingFaceH4/ultrafeedback_binarized  (61k AI-annotated pairs, higher quality)
"""

from datasets import load_dataset, Dataset
from typing import Tuple


# ── HH-RLHF ──────────────────────────────────────────────────────────────────

def load_hh_rlhf(
    split_ratio: float = 0.02,
    subset: str = "helpful-base",   # "helpful-base" | "harmless-base" | None (all)
) -> Tuple[Dataset, Dataset]:
    """
    Load Anthropic/hh-rlhf.
    Raw format: {"chosen": "Human: ...\n\nAssistant: ...", "rejected": "..."}

    subset="helpful-base" is recommended for a first run — cleaner signal,
    faster convergence than mixing helpful+harmless.
    """
    if subset:
        ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split="train")
    else:
        ds = load_dataset("Anthropic/hh-rlhf", split="train")

    split = ds.train_test_split(test_size=split_ratio, seed=42)
    train = split["train"].map(
        format_dpo_sample,
        remove_columns=split["train"].column_names,
        desc="Formatting train",
    )
    eval_ = split["test"].map(
        format_dpo_sample,
        remove_columns=split["test"].column_names,
        desc="Formatting eval",
    )

    # Drop examples where response extraction failed (empty string)
    train = train.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
    eval_ = eval_.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)

    return train, eval_


def _extract_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Split "Human: ...\n\nAssistant: ..." into (prompt, response).
    Prompt = everything up to and including the last "Assistant:".
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    prompt   = text[: idx + len(marker)]
    response = text[idx + len(marker):].strip()
    return prompt, response


def format_dpo_sample(example: dict) -> dict:
    prompt, chosen_resp   = _extract_prompt_and_response(example["chosen"])
    _,      rejected_resp = _extract_prompt_and_response(example["rejected"])
    return {
        "prompt":   prompt,
        "chosen":   chosen_resp,
        "rejected": rejected_resp,
    }


# ── UltraFeedback ─────────────────────────────────────────────────────────────

def load_ultrafeedback(split_ratio: float = 0.02) -> Tuple[Dataset, Dataset]:
    """
    HuggingFaceH4/ultrafeedback_binarized.
    Already binarized; chosen/rejected are lists of message dicts.
    Higher quality than HH-RLHF but requires slightly different formatting.
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    split = ds.train_test_split(test_size=split_ratio, seed=42)

    train = split["train"].map(
        format_ultrafeedback_sample,
        remove_columns=split["train"].column_names,
        desc="Formatting UltraFeedback train",
    )
    eval_ = split["test"].map(
        format_ultrafeedback_sample,
        remove_columns=split["test"].column_names,
        desc="Formatting UltraFeedback eval",
    )
    train = train.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
    eval_ = eval_.filter(lambda x: len(x["chosen"]) > 10 and len(x["rejected"]) > 10)
    return train, eval_


def format_ultrafeedback_sample(example: dict) -> dict:
    """
    UltraFeedback stores conversations as:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    We want: prompt = user turn, chosen/rejected = assistant turns.
    """
    def last_assistant(turns):
        for turn in reversed(turns):
            if turn.get("role") == "assistant":
                return turn.get("content", "")
        return ""

    # Build prompt from all non-final-assistant turns
    chosen_turns = example["chosen"]
    prompt_turns = [t for t in chosen_turns if t.get("role") != "assistant"]
    prompt = "\n".join(
        f"Human: {t['content']}" if t["role"] == "user" else f"Assistant: {t['content']}"
        for t in prompt_turns
    ) + "\n\nAssistant:"

    return {
        "prompt":   prompt,
        "chosen":   last_assistant(example["chosen"]),
        "rejected": last_assistant(example["rejected"]),
    }


# ── Inspect ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading HH-RLHF helpful-base...")
    train, val = load_hh_rlhf()
    s = train[0]
    print(f"Train: {len(train):,} | Val: {len(val):,}")
    print(f"\nPROMPT:   {s['prompt'][:200]}")
    print(f"CHOSEN:   {s['chosen'][:150]}")
    print(f"REJECTED: {s['rejected'][:150]}")
