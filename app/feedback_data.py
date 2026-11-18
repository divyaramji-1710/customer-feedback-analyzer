"""Feedback categories and reproducible dataset splits."""
import json
import random
from pathlib import Path

EXAMPLES = {
    "product_issue": [
        "The headphones stopped working after one day.",
        "The shirt arrived with a torn sleeve.",
        "My new kettle leaks water.",
        "The charger does not charge my phone.",
        "The shoes came with a broken sole.",
        "The screen flickers whenever I switch it on.",
        "The bag zipper broke on the first use.",
        "The bottle lid will not close properly.",
        "The speaker makes a crackling noise.",
        "The lamp arrived with a cracked base.",
    ],
    "staff_praise": [
        "The cashier was patient and friendly.",
        "Your assistant helped me find the right size.",
        "The manager handled my question politely.",
        "A staff member explained everything clearly.",
        "The checkout employee was very helpful.",
        "Your team made me feel welcome.",
        "The salesperson gave excellent advice.",
        "The employee patiently demonstrated the product.",
        "The assistant went out of her way to help.",
        "The staff treated me with kindness.",
    ],
    "delivery_issue": [
        "My parcel arrived three days late.",
        "The courier delivered to the wrong address.",
        "My order still has not arrived.",
        "The delivery driver left my parcel in the rain.",
        "The tracking says delivered but nothing arrived.",
        "The package was sent to another building.",
        "My express delivery missed the promised date.",
        "The courier never attempted delivery.",
        "The parcel arrived much later than expected.",
        "My order was lost during shipping.",
    ],
    "pricing_issue": [
        "I was charged more than the displayed price.",
        "The advertised discount was not applied.",
        "The receipt shows an unexpected extra charge.",
        "The shelf price differs from the checkout price.",
        "My coupon was not deducted from the total.",
        "I was billed twice for the same item.",
        "The sale price disappeared at checkout.",
        "The final bill includes an unexplained fee.",
        "The cashier charged the full price during the sale.",
        "The promotional offer was missing from my bill.",
    ],
    "return_refund": [
        "How do I return my purchase?",
        "I would like a refund for this order.",
        "Can I exchange this for another size?",
        "My refund has not reached my account.",
        "Please explain your return policy.",
        "Where can I get a return shipping label?",
        "I sent the item back but have not been refunded.",
        "Can I return this without the original packaging?",
        "Please help me arrange an exchange.",
        "When will my return be processed?",
    ],
}

def milestone_1(output_dir="data"):
    rng = random.Random(42)
    splits = {"train": [], "validation": [], "test": []}
    for label, comments in EXAMPLES.items():
        rows = [{"text": text, "label": label} for text in comments]
        rng.shuffle(rows)
        splits["train"].extend(rows[:6])
        splits["validation"].extend(rows[6:8])
        splits["test"].extend(rows[8:])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        rng.shuffle(rows)
        content = "".join(json.dumps(row) + "\n" for row in rows)
        (destination / f"{name}.jsonl").write_text(content, encoding="utf-8")
    return splits
