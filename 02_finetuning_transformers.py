"""
Topic 2: Generative AI Engineering and Fine-Tuning Transformers

Concept:
Demonstrate the complete Fine-Tuning process of a pre-trained
Hugging Face model to adapt it to a specific task (e.g., Text Classification)
using the `Trainer` abstraction and the `datasets` library.
"""

import sys

def main():
    print("--- 2. Generative AI Engineering and Fine-Tuning Transformers ---")
    print("Example: Fine-tuning DistilBERT for Sentiment Analysis.\n")

    try:
        # Importamos las dependencias necesarias
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer
        )
        import datasets

        print("[OK] The transformers, torch, and datasets libraries are installed.")

        # === 1. Model and Tokenizer Definition ===
        model_name = "distilbert-base-uncased"
        print(f"-> Preparing base model: {model_name}")

        print("-> Downloading model weights and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load the model indicating it will have 2 output labels (Positive/Negative)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        # === 2. Dataset Preparation ===
        print("-> Downloading and tokenizing the dataset (IMDb)...")
        # We use a very small dataset (100 examples) for demonstration; this way it runs fast on CPU.
        dataset = datasets.load_dataset("imdb", split="train[:100]")

        # Function to transform text into numerical model tokens
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Split into training set (80) and validation set (20)
        small_train_dataset = tokenized_datasets.select(range(80))
        small_eval_dataset = tokenized_datasets.select(range(80, 100))

        # === 3. Training Configuration (TrainingArguments) ===
        # Here we define the learning hyperparameters
        training_args = TrainingArguments(
            output_dir="./02_finetuning_transformers/model_results",
            eval_strategy="epoch",       # Evaluate at the end of each epoch
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,          # Only 1 epoch for demonstration
            weight_decay=0.01,
            push_to_hub=False,           # Do not upload to HuggingFace Hub
        )

        # === 4. Fine-Tuning Loop using Trainer ===
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
        )

        print("\n=== STARTING FINE-TUNING ===")
        print("Training the model on the adapted dataset (may take a minute)...")
        trainer.train()

        print("\\n=== EVALUATION ===")
        results = trainer.evaluate()
        print(f"Metric results: {results}")

        # === 5. Saving the fine-tuned model ===
        print("\\n-> Saving the adapted model locally...")
        trainer.save_model("./02_finetuning_transformers/my_finetuned_model")
        print("[Success] Fine-Tuning completed and model saved in './02_finetuning_transformers/my_finetuned_model'")

    except ImportError as e:
        print(f"[ERROR] Missing dependencies to run the Fine-Tuning environment: {e}")
        print("Make sure to run in terminal: pip install torch transformers datasets accelerate")

if __name__ == "__main__":
    main()
