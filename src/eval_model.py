from typing import List, Tuple

import torch
from datasets import Dataset
from peft import PeftModel, PeftMixedModel
from transformers import PreTrainedTokenizerBase, PreTrainedModel


def linearly_infer_from(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sample_results: Dataset) -> List[
    Tuple[str, str]]:
    """
    for every sample, we first tokenize the input (after applying the chat template) and then
    proceed to feed to model when then generates the output, and we then decode the output
    and save both the prompt message and the output

    it worked with the system with less GPU RAM but took 1.5 hrs more time than expected.
    :param model: model object (Fine-Tuned Model)
    :param tokenizer: Tokenizer Object
    :param sample_results: List of test sample's input.
    :return: List of Prompt Message, Output
    """
    model.eval()
    base_outputs = []
    for base_msg in sample_results:
        # we make sure to apply chat template to every message
        prompt = tokenizer.apply_chat_template(
            base_msg["messages"],
            tokenize=False,
            add_generation_prompt=True
        )

        # we tokenize the input and then make sure the model and inputs run on same device
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        with torch.no_grad():
            # we then provide this inputs
            # under the temperature (0.7)
            # top 0.9 (90%) probable results are only considered

            output = model.generate(
                **inputs,
                max_new_tokens=10_000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        base_outputs.append((base_msg, response))
        print(len(response))

    return base_outputs


def batch_infer_from(
        model: PeftModel | PeftMixedModel, tokenizer: PreTrainedTokenizerBase, sample_results: Dataset,
        batch: int = 4
) -> List[Tuple[str, str]]:
    """
        for every sample, we first tokenize the input (after applying the chat template) and then
        proceed to feed to model when then generates the output, and we then decode the output
        and save both the prompt message and the output

        the difference from previous function was that this function batches the inputs so it can
        process n inputs parallely in the expense of more RAM

        :param batch: number of inputs to process parallely
        :param model: model object (Fine-Tuned Model)
        :param tokenizer: Tokenizer Object
        :param sample_results: List of test sample's input.
        :return: List of Prompt Message, Output
    """
    model.eval()
    cs_outputs = []
    batch_size = batch  # increase if GPU allows
    total = len(sample_results)

    # Cache for speed
    apply_template = tokenizer.apply_chat_template
    device = model.device

    with torch.no_grad():
        for start in range(0, total, batch_size):
            batch = sample_results[start:start + batch_size]

            # Apply chat template
            prompts = [
                apply_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for msg in batch["messages"]
            ]

            # Tokenize with padding
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)

            # we then provide this inputs
            # under the temperature (0.7)
            # top 0.9 (90%) probable results are only considered
            outputs = model.generate(
                **inputs,
                max_new_tokens=10_000,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            # cache could save time here

            input_lens = inputs["input_ids"].shape[1]

            # Decode outputs
            for i, output in enumerate(outputs):
                response = tokenizer.decode(
                    output[input_lens:],
                    skip_special_tokens=True
                )
                cs_outputs.append((batch["messages"][i], response))

            progress = min(start + batch_size, total) / total * 100
            print(f"{progress:.2f}% complete")

    return cs_outputs


EXPORT_BASE_RESULTS = "base_outputs.json"
EXPORT_CS_RESULTS = "cs_outptus.json"
EXPORT_CS_TLDR_RESULTS = "cts_outputs.json"
