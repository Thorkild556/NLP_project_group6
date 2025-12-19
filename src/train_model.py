from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase
)
from typing import Any
from datasets import Dataset


def configure_trainer(output_dir: str, num_train_epochs=1) -> TrainingArguments:
    print("\n[6/8] Configuring training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,

        # Memory-optimized batch settings
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,

        # Learning settings
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=1.0,

        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,

        # Optimization
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,

        # Efficiency
        group_by_length=True,
        dataloader_num_workers=4,

        # Logging
        logging_steps=10,
        report_to="none",
    )
    return training_args


def print_args(train_dataset: Dataset, training_args: TrainingArguments) -> None:
    print("✓ Training configuration:")
    print(
        f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(
        f"  - Total training steps: ~{len(train_dataset) * 3 // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    print(f"  - Learning rate: {training_args.learning_rate}")


def print_train_progress(trainer: Trainer) -> Any:
    # Start training
    print("\n[8/8] Starting training...")
    print("=" * 60)
    print("TRAINING IN PROGRESS")
    print("=" * 60)

    return trainer.train()


EXPORT_TLDR_FINE_TUNED = "./llama3.2-3b-qlora-summary"
EXPORT_TLDR_CS_FINE_TUNED = "./final-summary"
EXPORT_CS_FINE_TUNED = "./llama3.2-3b-qlora-summary"

def export_model(trainer: Trainer, tokenizer: PreTrainedTokenizerBase, path: str) -> None:
    # Save final model
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nSaving model...")
    trainer.save_model(path)
    tokenizer.save_pretrained(path)
