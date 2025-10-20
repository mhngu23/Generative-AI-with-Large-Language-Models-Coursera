
from datasets import load_dataset

def prepare_data(dataset_name: str, split: str = "train"):
    """Load and optionally preprocess dataset."""
    print(f"📥 Loading dataset: {dataset_name} ({split})")
    if split is None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, split=split)
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset

def tokenize_function(tokenizer, example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example