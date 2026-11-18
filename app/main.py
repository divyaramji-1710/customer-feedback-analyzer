"""Streamlit interface for the locally fine-tuned feedback classifier."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.inference import FeedbackClassifier

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="💬")
st.title("💬 Customer Feedback Analyzer")
st.write("Categorize a customer comment using a locally fine-tuned GPT-2 model.")
st.caption(
    "Prototype trained on 30 synthetic comments. Validation accuracy: "
    "60% on 10 comments. Predictions can be wrong; review them before acting."
)

LABEL_NAMES = {
    "product_issue": "Product issue",
    "staff_praise": "Staff praise",
    "delivery_issue": "Delivery issue",
    "pricing_issue": "Pricing issue",
    "return_refund": "Return or refund",
}

@st.cache_resource
def load_classifier():
    return FeedbackClassifier(str(ROOT / "artifacts" / "feedback"))

with st.form("feedback_form"):
    comment = st.text_area(
        "Customer feedback",
        placeholder="For example: My parcel arrived three days late.",
        height=140,
    )
    submitted = st.form_submit_button("Classify feedback")

if submitted:
    if not comment.strip():
        st.warning("Enter a feedback comment first.")
    else:
        try:
            with st.spinner("Classifying your comment..."):
                classifier = load_classifier()
                result = classifier.classify(comment)

            if result["label"] == "unknown":
                st.warning(
                    "The model did not generate a recognized category. "
                    "Please review this comment manually."
                )
            else:
                st.subheader("Predicted category")
                st.write(LABEL_NAMES[result["label"]])

            if result["truncated"]:
                st.warning(
                    "This comment exceeded the model's input limit. "
                    "Only its beginning was classified."
                )
        except OSError:
            st.error(
                "Could not load the local model. Make sure training finished "
                "and artifacts/feedback contains the saved model and tokenizer."
            )

with st.expander("Available categories"):
    for name in LABEL_NAMES.values():
        st.write(f"- {name}")
