"""Checks that run without downloading or training a model."""
import json
from collections import Counter

from app.feedback_data import EXAMPLES, milestone_1
from app.encoding import MAX_LENGTH, encode_pair
from app.modeling import LABEL_TOKENS

def test_dataset_is_balanced_and_saved(tmp_path):
    splits = milestone_1(tmp_path)
    assert len(EXAMPLES) == 5
    for name, per_category in [("train", 6), ("validation", 2), ("test", 2)]:
        rows = splits[name]
        assert Counter(row["label"] for row in rows) == {
            label: per_category for label in EXAMPLES
        }
        saved = [
            json.loads(line)
            for line in (tmp_path / f"{name}.jsonl").read_text().splitlines()
        ]
        assert saved == rows

def test_splits_have_no_duplicate_comments(tmp_path):
    splits = milestone_1(tmp_path)
    texts = [
        row["text"].strip().lower()
        for rows in splits.values()
        for row in rows
    ]
    assert len(texts) == len(set(texts))

def test_dataset_is_reproducible(tmp_path):
    assert milestone_1(tmp_path / "a") == milestone_1(tmp_path / "b")

class StubTokenizer:
    """Controlled token lengths for testing encoding boundaries."""
    def __init__(self, length):
        self.length = length

    def encode(self, text, add_special_tokens=False):
        return [7] * self.length

    def convert_tokens_to_ids(self, token):
        assert token in LABEL_TOKENS.values()
        return 99

def test_only_category_token_is_supervised():
    encoded = encode_pair(
        StubTokenizer(12),
        {"text": "A damaged item.", "label": "product_issue"},
    )
    assert encoded["labels"] == [-100] * 12 + [99]
    assert encoded["input_ids"][-1] == 99
    assert encoded["attention_mask"] == [1] * 13

def test_long_input_keeps_category_target():
    encoded = encode_pair(
        StubTokenizer(MAX_LENGTH * 2),
        {"text": "A long comment.", "label": "delivery_issue"},
    )
    assert len(encoded["input_ids"]) == MAX_LENGTH
    assert len(encoded["labels"]) == MAX_LENGTH
    assert encoded["labels"][-1] == 99
    assert sum(value != -100 for value in encoded["labels"]) == 1
