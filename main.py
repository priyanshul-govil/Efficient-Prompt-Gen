# main.py
import argparse
import torch
from dataset import load_and_preprocess_dataset
from model import load_model_and_tokenizer
from train import train_prompt_tuning
from eval import evaluate_prompt_tuning, evaluate_with_logits
from transformers import AdamW, get_linear_schedule_with_warmup
from config import LEARNING_RATE, NUM_EPOCHS, OUTPUT_DIR
import os
os.environ["HF_HOME"] = "~/ada_user/.cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Prompt-tune and evaluate models on multiple datasets.")
    parser.add_argument("--dataset", type=str, default="sst2", choices=["sst2", "nli"], help="Dataset to use")
    parser.add_argument("--model", type=str, default="bloomz", choices=["bloomz", "gpt2", "llama"], help="Model to use")
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name=args.model)
    model.to(device)

    # Preprocess dataset
    dataset_name = args.dataset
    ds = load_and_preprocess_dataset(dataset_name=dataset_name, tokenizer=tokenizer)

    # Split datasets
    train_ds = ds["train"].select(range(10000))
    eval_ds = ds["validation"].select(range(500))

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=(len(train_ds) * NUM_EPOCHS)
    )

    # print("Test set column names:", ds["test"].column_names)

    model_path = f"..//ada_user/{OUTPUT_DIR}/{args.dataset}_bmodel"

    torch.save(model.state_dict(), model_path)
    
    ## Evaluate the model
    # evaluate_with_logits(model, tokenizer, ds, dataset_name, device)

    print("\n\n Training the model...\n\n")
    # Train the model
    train_prompt_tuning(model, train_ds, eval_ds, optimizer, scheduler, device, dataset_name, model_path)

    torch.save(model.state_dict(), model_path)

    print(model_path)
    # Evaluate the model
    evaluate_with_logits(model, tokenizer, ds, dataset_name, device)

if __name__ == "__main__":
    main()
