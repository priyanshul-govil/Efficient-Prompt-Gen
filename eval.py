# eval.py
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import torch

def evaluate_prompt_tuning(model_path, tokenizer, ds, dataset_name, device):
    """
    Evaluate a prompt-tuned model on a dataset.

    Args:
        model_path (str): Path to the fine-tuned model.
        tokenizer: Tokenizer instance.
        ds: The dataset to evaluate.
        dataset_name (str): Name of the dataset (e.g., "sst2", "nli").
        device: Torch device (e.g., "cuda" or "cpu").
    """
    # Load the model
    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    # Dataset-specific column settings
    text_column = "sentence" if dataset_name == "sst2" else "premise"
    label_column = "label"

    # Initialize counters for accuracy calculation
    correct_predictions = 0
    total_samples = len(ds["validation"])

    # Progress bar for evaluation
    for i in tqdm(range(total_samples), desc=f"Evaluating {dataset_name}"):
        # Prepare the input text
        if dataset_name == "sst2":
            input_text = f'Sentence: {ds["validation"][i][text_column]} Sentiment(postive/negative): '
        elif dataset_name == "nli":
            input_text = f'Premise: {ds["validation"][i]["premise"]} Hypothesis: {ds["validation"][i]["hypothesis"]} Relation: '
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)

        # Decode the output and compare with the ground truth label
        output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        ground_truth_label = str(ds["validation"][i][label_column])
        print(output, ground_truth_label)
        
        if output.split()[-1].lower() == "negative":
            predicted_label = "0"
        elif output.split()[-1].lower() == "positive":
            predicted_label = "1"
        else:
            predicted_label = "-1"
        
        if predicted_label == str(ground_truth_label):
            correct_predictions += 1

    # Calculate and print accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy on {dataset_name}: {accuracy * 100:.2f}%")

from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import torch
import torch.nn.functional as F

def evaluate_with_logits(model_path, tokenizer, ds, dataset_name, device):
    """
    Evaluate a prompt-tuned model by comparing logits for specific tokens.

    Args:
        model_path (str): Path to the fine-tuned model.
        tokenizer: Tokenizer instance.
        ds: The dataset to evaluate.
        dataset_name (str): Name of the dataset (e.g., "sst2", "nli").
        device: Torch device (e.g., "cuda" or "cpu").
    """
    # Load the model
    model = AutoPeftModelForCausalLM.from_pretrained(model_path)
    model.to(device)

    # Token IDs for "positive" and "negative"
    positive_token_id = tokenizer("positive", add_special_tokens=False)["input_ids"][0]
    negative_token_id = tokenizer("negative", add_special_tokens=False)["input_ids"][0]

    # Dataset-specific column settings
    text_column = "sentence" if dataset_name == "sst2" else "premise"
    label_column = "label"

    # Initialize counters for accuracy calculation
    correct_predictions = 0
    total_samples = len(ds["validation"])

    # Progress bar for evaluation
    for i in tqdm(range(total_samples), desc=f"Evaluating {dataset_name}"):
        # Prepare the input text
        if dataset_name == "sst2":
            input_text = f'Sentence: {ds["validation"][i][text_column]} Sentiment: '
        elif dataset_name == "nli":
            input_text = f'Premise: {ds["validation"][i]["premise"]} Hypothesis: {ds["validation"][i]["hypothesis"]} Relation: '
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract logits for the last token position
        logits = outputs.logits[:, -1, :]  # Shape: (batch_size=1, vocab_size)

        # Compute probabilities for "positive" and "negative"
        positive_prob = F.softmax(logits, dim=-1)[0, positive_token_id].item()
        negative_prob = F.softmax(logits, dim=-1)[0, negative_token_id].item()

        # Predict based on higher probability
        predicted_label = "positive" if positive_prob > negative_prob else "negative"
        # print(positive_prob, negative_prob)

        # Get the ground truth label
        ground_truth_label = ds["validation"][i][label_column]
        ground_truth_label = "positive" if ground_truth_label == 1 else "negative"  # Adjust for dataset format
        # print(predicted_label, ground_truth_label)
        # Compare predicted and ground truth labels
        if predicted_label == ground_truth_label:
            correct_predictions += 1

    # Calculate and print accuracy
    accuracy = correct_predictions / total_samples
    print(f"Accuracy on {dataset_name}: {accuracy * 100:.2f}%")
