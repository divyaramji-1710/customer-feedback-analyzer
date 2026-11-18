"""Load GPT-2 with one special token per feedback category."""
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.feedback_data import EXAMPLES

MODEL_NAME = "distilgpt2"
LABEL_TOKENS = {label: f"<{label}>" for label in EXAMPLES}

def prepare_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    added = tokenizer.add_special_tokens({
        "additional_special_tokens": list(LABEL_TOKENS.values())
    })

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    assert model.get_input_embeddings().num_embeddings == len(tokenizer)

    print("Added category tokens:", added)
    print("Tokenizer size:", len(tokenizer))
    print("Embedding rows:", model.get_input_embeddings().num_embeddings)
    for label, token in LABEL_TOKENS.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        assert len(ids) == 1
        print(f"{label}: {token} -> {ids}")

    return model, tokenizer
