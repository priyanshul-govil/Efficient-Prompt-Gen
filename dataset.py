# dataset.py
from datasets import load_dataset
from config import DATASET_CONFIGS, MAX_LENGTH


def preprocess_sst2(examples, tokenizer):
    prompt_template = DATASET_CONFIGS["sst2"]["prompt_template"]
    inputs = [prompt_template.format(sentence=x) for x in examples["sentence"]]
    targets = [str(x) for x in examples["text_label"]]
    tokenized = _tokenize(inputs, targets, tokenizer)
    tokenized["sentence"] = examples[
        "sentence"
    ]  # Retain original column for evaluation
    return tokenized


def preprocess_nli(examples, tokenizer):
    prompt_template = DATASET_CONFIGS["nli"]["prompt_template"]
    inputs = [
        prompt_template.format(premise=p, hypothesis=h)
        for p, h in zip(examples["premise"], examples["hypothesis"])
    ]
    targets = [str(x) for x in examples["text_label"]]
    tokenized = _tokenize(inputs, targets, tokenizer)
    tokenized["premise"] = examples["premise"]  # Retain original column for evaluation
    tokenized["hypothesis"] = examples[
        "hypothesis"
    ]  # Retain original column for evaluation
    return tokenized


def _tokenize(inputs, targets, tokenizer):
    model_inputs = tokenizer(
        inputs, padding="max_length", max_length=MAX_LENGTH, truncation=True
    )
    labels = tokenizer(targets, padding=False, truncation=True)

    for i in range(len(inputs)):
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            MAX_LENGTH - len(model_inputs["input_ids"][i])
        ) + model_inputs["input_ids"][i]
        model_inputs["attention_mask"][i] = [0] * (
            MAX_LENGTH - len(model_inputs["attention_mask"][i])
        ) + model_inputs["attention_mask"][i]
        labels["input_ids"][i] = [-100] * (
            MAX_LENGTH - len(labels["input_ids"][i])
        ) + labels["input_ids"][i]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_and_preprocess_dataset(dataset_name="sst2", tokenizer=None):
    config = DATASET_CONFIGS[dataset_name]
    ds = load_dataset(config["name"], config["config"])

    # Add the `text_label` column dynamically
    classes = config["classes"]
    ds = ds.map(
        lambda batch: {"text_label": [classes[label] for label in batch["label"]]},
        batched=True,
        num_proc=1,
        desc="Mapping text labels to numeric labels",
    )

    # Preprocess examples
    preprocess_function = lambda examples: (
        preprocess_sst2(examples, tokenizer)
        if dataset_name == "sst2"
        else preprocess_nli(examples, tokenizer)
    )
    processed_ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        desc=f"Running tokenizer on {dataset_name} dataset",
    )
    return processed_ds
