#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main entry script for training or inference.
Supports both full fine-tuning and PEFT (LoRA) fine-tuning.
"""
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch

from prepare_data import prepare_data, tokenize_function
from train import train_full, train_peft
from utils import load_model_and_tokenizer, print_number_of_trainable_model_parameters

print("Torch CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Torch built with CUDA:", torch.backends.cudnn.is_available())

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train model with full or PEFT fine-tuning")

    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "peft"],
        help="Training mode: 'full' for full fine-tuning, 'peft' for parameter-efficient fine-tuning (LoRA).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="knkarthick/dialogsum",
        help="Name of the dataset to load from Hugging Face.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-small",
        help="Pretrained model name or path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/s223669184/llm_training/results",
        help="Directory to save model checkpoints and logs.",
    )

    return parser.parse_args()


def main():
    """Main function for training or inference."""
    start_time = time.time()
    args = parse_args()

    print(f"🚀 Starting training in '{args.mode}' mode...")

    base_dir = "/scratch/s223669184/llm_training/results"
    if args.mode == "full":
        args.output_dir = os.path.join(base_dir, "full")
    elif args.mode == "peft":
        args.output_dir = os.path.join(base_dir, "peft")
    else:
        args.output_dir = os.path.join(base_dir, "other")

    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model_name}")

    # --- Load dataset ---
    dataset = prepare_data(args.dataset_name, split=None)

    # --- Load model and tokenizer ---
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # --- Print model parameters ---
    print(print_number_of_trainable_model_parameters(model))

    # --- Tokenize dataset ---
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(tokenizer, examples),
        batched=True
    )

    tokenized_datasets = tokenized_datasets.remove_columns(
        ["id", "topic", "dialogue", "summary"]
    )

    # --- Train ---
    if args.mode == "peft":
        print("🧩 Using PEFT (LoRA) fine-tuning.")
        train_peft(model, tokenizer, tokenized_datasets, output_dir=args.output_dir)
    else:
        print("🧠 Using full fine-tuning.")
        train_full(model, tokenizer, tokenized_datasets, output_dir=args.output_dir)

    print(f"🏁 Finished in {(time.time() - start_time):.2f} seconds.")


if __name__ == "__main__":
    main()
