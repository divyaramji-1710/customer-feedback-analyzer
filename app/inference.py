"""Classify feedback using the saved fine-tuned model."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.encoding import MAX_LENGTH
from app.modeling import LABEL_TOKENS

class FeedbackClassifier:
    def __init__(self, model_dir="artifacts/feedback"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()
        self.id_to_label = {
            self.tokenizer.convert_tokens_to_ids(token): label
            for label, token in LABEL_TOKENS.items()
        }

    @torch.inference_mode()
    def classify(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a non-empty feedback comment.")

        prompt = f"Feedback: {text.strip()}\nCategory:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        truncated = len(ids) > MAX_LENGTH - 1
        ids = ids[:MAX_LENGTH - 1]
        inputs = torch.tensor([ids], dtype=torch.long)

        output = self.model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        token_id = output[0, -1].item()
        return {
            "label": self.id_to_label.get(token_id, "unknown"),
            "token": self.tokenizer.decode([token_id]),
            "truncated": truncated,
        }

def evaluate(model_dir="artifacts/feedback", split="validation"):
    if split not in {"validation", "test"}:
        raise ValueError("Choose validation or test.")

    classifier = FeedbackClassifier(model_dir)
    path = Path("data") / f"{split}.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    predictions = []
    for row in rows:
        result = classifier.classify(row["text"])
        correct = result["label"] == row["label"]
        predictions.append({
            **row,
            "predicted": result["label"],
            "generated_token": result["token"],
            "correct": correct,
        })
        print(f"TRUE: {row['label']:15} PRED: {result['label']:15} | {row['text']}")

    report = {
        "split": split,
        "examples": len(rows),
        "accuracy": sum(row["correct"] for row in predictions) / len(rows),
        "uniform_random_baseline": 1 / len(LABEL_TOKENS),
        "predictions": predictions,
    }
    destination = Path("reports")
    destination.mkdir(exist_ok=True)
    (destination / f"{split}_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nAccuracy: {report['accuracy']:.1%}")
    print("Uniform random baseline: 20%")
    print(f"Saved reports/{split}_evaluation.json")
    return report
