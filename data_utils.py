"""
data_utils.py
-------------
Load and format Anthropic HH-RLHF (or UltraFeedback) into
the triplet format DPOTrainer expects:
    {"prompt": str, "chosen": str, "rejected": str}
"""

from datasets import load_dataset, DatasetDict, Dataset
from typing import Tuple


# ── HH-RLHF ──────────────────────────────────────────────────────────────────

def load_hh_rlhf(split_ratio: float = 0.02) -> Tuple[Dataset, Dataset]:
    """
    Load Anthropic/hh-rlhf from HuggingFace.
    Returns (train_dataset, eval_dataset).

    The raw data looks like:
        {"chosen":   "Human: ...\n\nAssistant: ...",
         "rejected": "Human: ...\n\nAssistant: ..."}
    """
    ds = load_dataset("Anthropic/hh-rlhf", split="train")

    # Small eval split
    split = ds.train_test_split(test_size=split_ratio, seed=42)
    return split["train"], split["test"]


def _extract_prompt_and_response(text: str) -> Tuple[str, str]:
    """
    Split 'Human: ...\n\nAssistant: ...' into (prompt, response).
    The prompt is everything up to (and including) the last 'Assistant:'.
    """
    # Find last occurrence of "\n\nAssistant:"
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return text, ""
    prompt   = text[: idx + len(marker)]
    response = text[idx + len(marker):].strip()
    return prompt, response


def format_dpo_sample(example: dict) -> dict:
    """
    Map a raw HH-RLHF example to DPOTrainer format.
    Both chosen and rejected share the same prompt.
    """
    prompt, chosen_response   = _extract_prompt_and_response(example["chosen"])
    _,      rejected_response = _extract_prompt_and_response(example["rejected"])

    return {
        "prompt":   prompt,
        "chosen":   chosen_response,
        "rejected": rejected_response,
    }


# ── UltraFeedback (alternative dataset) ──────────────────────────────────────

def load_ultrafeedback(split_ratio: float = 0.02) -> Tuple[Dataset, Dataset]:
    """
    HuggingFaceH4/ultrafeedback_binarized already provides binarized pairs.
    Columns: prompt, chosen (list of dicts), rejected (list of dicts)
    """
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    split = ds.train_test_split(test_size=split_ratio, seed=42)
    return split["train"], split["test"]


def format_ultrafeedback_sample(example: dict) -> dict:
    """
    UltraFeedback stores chosen/rejected as conversation lists.
    Extract the last assistant turn as the response.
    """
    def last_assistant(turns):
        for turn in reversed(turns):
            if turn["role"] == "assistant":
                return turn["content"]
        return ""

    # Build prompt from all turns except the last assistant
    prompt_turns = [t for t in example["chosen"][:-1]]
    prompt = "\n".join(
        f"{'Human' if t['role']=='user' else 'Assistant'}: {t['content']}"
        for t in prompt_turns
    ) + "\n\nAssistant:"

    return {
        "prompt":   prompt,
        "chosen":   last_assistant(example["chosen"]),
        "rejected": last_assistant(example["rejected"]),
    }


# ── Quick dataset inspection ──────────────────────────────────────────────────

if __name__ == "__main__":
    train, val = load_hh_rlhf()
    sample = format_dpo_sample(train[0])
    print("=== Sample ===")
    print("PROMPT:\n",   sample["prompt"][:300])
    print("CHOSEN:\n",   sample["chosen"][:200])
    print("REJECTED:\n", sample["rejected"][:200])
    print(f"\nTrain size: {len(train):,} | Val size: {len(val):,}")
