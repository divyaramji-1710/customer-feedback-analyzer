"""Project milestone entry points."""
from app.feedback_data import milestone_1
from app.modeling import prepare_model

def milestone_2():
    return prepare_model()

def milestone_3(**kwargs):
    from app.training import train_model
    return train_model(**kwargs)

def milestone_4(text, model_dir="artifacts/feedback"):
    from app.inference import FeedbackClassifier
    return FeedbackClassifier(model_dir).classify(text)

def milestone_5():
    """Return the Streamlit entry point."""
    from pathlib import Path
    return Path(__file__).resolve().parent / "main.py"
