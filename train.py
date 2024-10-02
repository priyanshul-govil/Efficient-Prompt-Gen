# train.py
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import default_data_collator
from eval import evaluate_prompt_tuning, evaluate_with_logits
import os

from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, OUTPUT_DIR

def train_prompt_tuning(model, train_ds, eval_ds, optimizer, scheduler, device, dataset_name, model_path):
    """
    Train a prompt-tuned model.

    Args:
        model: The model to train.
        train_ds: Training dataset.
        eval_ds: Evaluation dataset.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Torch device (e.g., "cuda" or "cpu").
        dataset_name (str): The name of the dataset (e.g., "sst2").
    """
    train_dataloader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
    )
    eval_dataloader = DataLoader(
        eval_ds,
        collate_fn=default_data_collator,
        batch_size=BATCH_SIZE,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch} on {dataset_name}"):
            # Filter batch to include only required keys
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_dataloader)}")

        # Save model
        model.save_pretrained(model_path)

        evaluate_with_logits(model_path, tokenizer, ds, dataset_name, device)

