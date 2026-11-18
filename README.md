# Customer Feedback Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gbw8cuw9ms/badge.svg)](https://ask.thili.ai/b/gbw8cuw9ms)

This Customer Feedback Analyzer leverages a fine-tuned GPT-2 model to automatically categorize customer feedback. By generating special label tokens like `<product_issue>` or `<staff_praise>`, the tool helps retail managers quickly understand customer sentiment and focus on key areas for improvement.

## What you are building

**Goal.** You will build a tool that automatically categorizes customer feedback comments, helping retail managers quickly identify trends and prioritize improvements.

**Done when.** You will have a fine-tuned GPT-2 model capable of classifying new customer comments by generating special label tokens, and a Streamlit application where users can input comments and see their classifications.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Define 3-5 feedback categories and create a small dataset of example comments with labels.**  
      Hint: Define your categories and create a small dataset of example comments with labels, similar to how legal clauses were tagged.
- [ ] **2. Extend GPT-2 tokenizer and model with new category tokens.**  
      Hint: Extend the tokenizer and model with your new category tokens, as shown in the drill for legal tags.
- [ ] **3. Fine-tune the model using the Hugging Face Trainer.**  
      Hint: Use the Hugging Face Trainer, following the fine-tuning process from the drill.
- [ ] **4. Implement an inference function to classify new feedback.**  
      Hint: Implement an inference function to classify new feedback, similar to how predictions were made in the drill.
- [ ] **5. Build a simple Streamlit app to input comments and show classifications.**  
      Hint: Build a simple Streamlit app to input comments and show classifications, integrating your inference function.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset with more diverse examples for existing categories or add new feedback categories.
- Implement a feature to display the model's confidence score for each classification.
- Allow users to edit incorrect classifications and use that feedback to improve the model (e.g., for future re-training).
