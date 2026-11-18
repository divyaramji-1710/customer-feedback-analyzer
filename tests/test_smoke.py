"""Smoke checks for the local feedback app; no model download required."""
import py_compile
from pathlib import Path

import pytest

from app.core import milestone_5
from app.inference import FeedbackClassifier

def test_streamlit_entry_point_exists_and_compiles():
    page = milestone_5()
    assert page.is_file()
    py_compile.compile(str(page), doraise=True)

@pytest.mark.parametrize("text", ["", "   ", None])
def test_classifier_rejects_empty_input_without_loading_model(text):
    classifier = FeedbackClassifier.__new__(FeedbackClassifier)
    with pytest.raises(ValueError, match="non-empty"):
        classifier.classify(text)
