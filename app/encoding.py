"""Encode feedback while preserving its supervised category token."""
from app.modeling import LABEL_TOKENS

MAX_LENGTH = 128

def encode_pair(tokenizer, row):
    prompt = f"Feedback: {row['text']}\nCategory:"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    label_id = tokenizer.convert_tokens_to_ids(LABEL_TOKENS[row["label"]])

    # Reserve one position for the category, even for long comments.
    prompt_ids = prompt_ids[:MAX_LENGTH - 1]
    input_ids = prompt_ids + [label_id]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + [label_id],
    }
