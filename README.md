\# Customer Feedback Analyzer



A local Streamlit app that categorizes retail feedback using a fully

fine-tuned distilgpt2 model.



\## Categories



\- Product issue

\- Staff praise

\- Delivery issue

\- Pricing issue

\- Return or refund



\## How it works



Five special category tokens are added to the tokenizer, and the model's

embeddings are resized to match. Training supervises only the category

token following each feedback comment.



Hugging Face Trainer updates all 81,916,416 parameters. Inference generates

one token using greedy decoding. Unrecognized outputs are shown as unknown.



\## Setup



Requires Python 3.12 and Git. No GPU or API key is required.

Internet access is needed for the initial model download.



Run these commands in PowerShell from the project folder:



```powershell

py -3.12 -m venv .venv

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```



\## Generate data and train



```powershell

.\\.venv\\Scripts\\python.exe -c "from app.core import milestone\_1; milestone\_1()"

.\\.venv\\Scripts\\python.exe -u -c "from app.core import milestone\_3; milestone\_3()"

```



Training saves the model and tokenizer in artifacts/feedback.

Model weights and checkpoints are excluded from Git.



\## Evaluate



```powershell

.\\.venv\\Scripts\\python.exe -u -c "from app.inference import evaluate; evaluate(split='validation')"

.\\.venv\\Scripts\\python.exe -u -c "from app.inference import evaluate; evaluate(split='test')"

```



\## Launch the app



Train the model first, then run:



```powershell

.\\.venv\\Scripts\\python.exe -m streamlit run app/main.py

```



No Ollama, external chat API, or .env configuration is needed.



\## Run tests



```powershell

.\\.venv\\Scripts\\python.exe -m pytest -q

```



The automated tests do not download or train a model.



\## Recorded results



| Measure | Result |

| --- | --- |

| Synthetic comments | 50 across five balanced categories |

| Training / validation / test | 30 / 10 / 10 |

| Training epochs | 12 |

| Initial validation loss | 3.2696 |

| Best validation loss | 1.2433 |

| Validation accuracy | 60% (6/10) |

| Test accuracy | 60% (6/10) |

| Expected uniform random accuracy | 20% |

| Automated tests | 9 passed |



Evaluation reports and a copy of the training report are in reports/.

New training runs save their report under artifacts/feedback.



\## Limitations



This is a learning prototype. It uses only six training examples per

category, and each evaluation set contains ten comments. One prediction

therefore changes accuracy by ten percentage points.



The headphones test comment was used in a manual interface check before

the final test evaluation. The test set was therefore not completely

unseen during manual checks. No model changes were made based on that check.



Returns and refunds were confused with pricing and delivery issues.

Real comments may contain several issues or fall outside these categories.

The model predicts one category only. Long inputs are truncated to the

127-token prompt allowance, and the interface displays a warning.



The random baseline is theoretical rather than a separately tested model.

Interface examples demonstrate operation, not general accuracy.

More varied training data and a fresh evaluation set are needed before

claiming reliable performance.



\## Completed milestones



1\. Created five categories and reproducible labelled dataset splits.

2\. Added category tokens and resized model embeddings.

3\. Fully fine-tuned distilgpt2 using Hugging Face Trainer.

4\. Implemented inference and measured classification accuracy.

5\. Built a Streamlit interface with empty-input handling and limitations.



Built from the askThili starter and course approach with AI coding assistance.


