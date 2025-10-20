from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def load_model_and_tokenizer(model_name: str):
    """Load pretrained model and tokenizer."""
    print(f"⚙️ Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, tokenizer

def print_number_of_trainable_model_parameters(model):
    """Utility to count and report number of trainable parameters."""
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_model_params / all_model_params
    return (
        f"trainable model parameters: {trainable_model_params}\n"
        f"all model parameters: {all_model_params}\n"
        f"percentage of trainable model parameters: {percentage:.2f}%"
    )