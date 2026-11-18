"""Full fine-tuning with category-only supervision."""
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, set_seed

from app.encoding import encode_pair
from app.modeling import prepare_model

def train_model(output_dir="artifacts/feedback", epochs=12):
    set_seed(42)
    model, tokenizer = prepare_model()

    def load_split(name):
        path = Path("data") / f"{name}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        return Dataset.from_list([encode_pair(tokenizer, row) for row in rows])

    def collate(rows):
        longest = max(len(row["input_ids"]) for row in rows)
        batch = {}
        for key in ("input_ids", "attention_mask", "labels"):
            padding = {
                "input_ids": tokenizer.pad_token_id,
                "attention_mask": 0,
                "labels": -100,
            }[key]
            batch[key] = torch.tensor([
                row[key] + [padding] * (longest - len(row[key]))
                for row in rows
            ], dtype=torch.long)
        return batch

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(destination / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="none",
        dataloader_pin_memory=False,
        use_cpu=True,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=load_split("train"),
        eval_dataset=load_split("validation"),
        data_collator=collate,
    )

    print("Training examples:", len(trainer.train_dataset))
    print("Validation examples:", len(trainer.eval_dataset))
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    before = trainer.evaluate()
    print("Initial validation loss:", before["eval_loss"])

    trainer.train()
    after = trainer.evaluate()

    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))

    report = {
        "initial_validation_loss": before["eval_loss"],
        "best_validation_loss": after["eval_loss"],
        "train_examples": len(trainer.train_dataset),
        "validation_examples": len(trainer.eval_dataset),
        "epochs": epochs,
        "history": trainer.state.log_history,
    }
    (destination / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("Best validation loss:", after["eval_loss"])
    print("Saved model, tokenizer and report in", destination)
    return report
