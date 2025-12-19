from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel, PeftMixedModel
import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import PreTrainedTokenizerBase, BatchEncoding, PreTrainedModel
from typing import Optional, Callable
from datasets import Dataset

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def load_tokenizer(model_name=BASE_MODEL) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    # LLaMA models usually do not have a pad token by default
    tokenizer.pad_token = tokenizer.eos_token
    if model_name != BASE_MODEL:
        tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name=BASE_MODEL) -> PreTrainedModel:
    print("\n[2/8] Configuring 8-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
    print("\n[3/8] Loading LLaMA 3.2 3B model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("✓ Model loaded in 8-bit")
    print("✓ Model size: ~3B parameters")

    return model


def lora_config_for(model: PreTrainedModel, pretrained_path: Optional[str] = None, for_training=True) -> PeftModel | PeftMixedModel:
    was_pretrained = pretrained_path is not None

    print("\n[4/8] Preparing model for QLoRA...")
    # preparing the model for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )

    if was_pretrained:
        model = PeftModel.from_pretrained(
            model,
            pretrained_path,
            is_trainable=for_training  # to continue training
        )
        model.train()

        print("✓ Fine-tuned model loaded")

    # (Optional but recommended) disable cache during training
    model.config.use_cache = False

    if was_pretrained:
        return model

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    print("✓ LoRA adapters added")

    return model


# max token limit for the custom dataset
TOKEN_LIMIT_FOR_CS = 10_000


# helper function to tokenize our input and also apply LLaMA chat template
def apply_formatter(_tokenizer: PreTrainedTokenizerBase, token_limit: Optional[int] = 4096) -> Callable[
    [dict], BatchEncoding]:
    def format_and_tokenize(tokenizer: PreTrainedTokenizerBase, example: dict,
                            force: Optional[int] = 4096) -> BatchEncoding:
        # Apply LLaMA chat template
        formatted_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize (NO padding, NO manual labels)
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=force,  # IMPORTANT: increase for long transcripts
            padding=False,
        )

        return tokenized

    return lambda ex: format_and_tokenize(_tokenizer, ex, token_limit)


def format_dataset(formatter, dataset: Dataset, desc: str):
    print(desc)
    return dataset.map(
        formatter,
        remove_columns=["messages"],
        batched=False,
        desc=desc
    )


def prep_data_collector(tokenizer: PreTrainedTokenizerBase) -> DataCollatorForLanguageModeling:
    # Data collator with dynamic padding
    print("\n[7/8] Creating data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Efficient padding for GPU
    )

    return data_collator
