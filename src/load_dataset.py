from pathlib import Path
from typing import Union
import json
from datasets import Dataset, DatasetDict


def load_jsonl(file_path: Union[str, Path]) -> Dataset:
    """
    Load Dataset from the provided json file path line by line and return the Dataset
    :param file_path: path of the jsonl file
    :return: Dataset
    """
    # Load JSONL data (TL;DR Dataset)
    print("\n[1/8] Loading dataset...")
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    dataset = Dataset.from_dict({
        "messages": [item["messages"] for item in data]
    })

    print(f"✓ Loaded {len(dataset)} examples")
    return dataset


def split(dataset: Dataset) -> DatasetDict:
    _split = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✓ Train: {len(_split['train'])} | Val: {len(_split['test'])}")
    return _split


def print_sample(train_dataset) -> None:
    # Show sample stats (pre-training)
    sample = train_dataset[0]

    print("\n✓ Sample stats:")
    print(f"  - Input length: {len(sample['input_ids'])} tokens")
    print(f"  - Attention tokens: {sum(sample['attention_mask'])} tokens")
    print(f"  - Truncated: {'Yes' if len(sample['input_ids']) == 4096 else 'No'}")


CS_JSON = "custom_dataset.jsonl"
TL_DR_JSON = "proc_tldr.jsonl"