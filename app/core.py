"""Project milestone entry points."""
from app.feedback_data import milestone_1
from app.modeling import prepare_model

def milestone_2():
    return prepare_model()

def milestone_3(**kwargs):
    from app.training import train_model
    return train_model(**kwargs)

def milestone_4():
    raise NotImplementedError("Milestone 4 is next.")

def milestone_5():
    raise NotImplementedError("Milestone 5 is next.")
