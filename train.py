from transformers import TrainingArguments, Trainer
import evaluate
from peft import LoraConfig, get_peft_model, TaskType

from utils import print_number_of_trainable_model_parameters

def train_full(model, tokenizer, dataset, output_dir="./results"):


    """Train model using Hugging Face Trainer."""
    print("🚀 Starting training...")

    training_args = TrainingArguments(
        output_dir=output_dir,              # Directory where model checkpoints and the final model will be saved
        evaluation_strategy="epoch",        # When to run evaluation; here, after each epoch
        save_strategy="epoch",              # When to save checkpoints; here, after each epoch
        learning_rate=2e-5,                 # Initial learning rate for the optimizer
        per_device_train_batch_size=1,      # Batch size per device (GPU/CPU) for training
        per_device_eval_batch_size=1,       # Batch size per device (GPU/CPU) for evaluation
        num_train_epochs=3,                 # Number of full passes over the training dataset
        weight_decay=0.01,                  # Weight decay for regularization to prevent overfitting
        logging_dir="/scratch/s223669184/llm_training/logs/full",               # Directory to save TensorBoard logs
        logging_steps=10,                   # Frequency (in steps) of logging training metrics
        save_total_limit=2,                 # Maximum number of saved checkpoints; older ones will be deleted
        push_to_hub=False,                  # Whether to push model checkpoints to Hugging Face Hub
    )

    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: v.mid.fmeasure for k, v in result.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("✅ Training complete!")

def train_peft(model, tokenizer, dataset, output_dir="./results"):
    """Train model using PEFT (LoRA) with Hugging Face Trainer."""
    print("🚀 Starting PEFT (LoRA) training...")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    peft_model = get_peft_model(model, lora_config)

    print(print_number_of_trainable_model_parameters(peft_model))

    training_args = TrainingArguments(
        output_dir=output_dir,              # Directory where model checkpoints and the final model will be saved
        evaluation_strategy="epoch",        # When to run evaluation; here, after each epoch
        save_strategy="epoch",              # When to save checkpoints; here, after each epoch
        learning_rate=2e-5,                 # Initial learning rate for the optimizer
        per_device_train_batch_size=1,      # Batch size per device (GPU/CPU) for training
        per_device_eval_batch_size=1,       # Batch size per device (GPU/CPU) for evaluation
        num_train_epochs=3,                 # Number of full passes over the training dataset
        weight_decay=0.01,                  # Weight decay for regularization to prevent overfitting
        logging_dir="/scratch/s223669184/llm_training/logs/peft",               # Directory to save TensorBoard logs
        logging_steps=10,                   # Frequency (in steps) of logging training metrics
        save_total_limit=2,                 # Maximum number of saved checkpoints; older ones will be deleted
        push_to_hub=False,                  # Whether to push model checkpoints to Hugging Face Hub
    )

    metric = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {k: v.mid.fmeasure for k, v in result.items()}

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("✅ PEFT (LoRA) training complete!")
