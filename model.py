# model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from config import MODEL_CONFIGS, NUM_VIRTUAL_TOKENS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(model_name="bloomz"):
    if model_name == "llama":
        config = MODEL_CONFIGS[model_name]
        model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token="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",
        device_map=device,
        )
    
        tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        use_fast=False,
        use_auth_token="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",
        )
        tokenizer.apply_chat_template
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        config = MODEL_CONFIGS[model_name]
        model = AutoModelForCausalLM.from_pretrained(config["model_name"])
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Setup Prompt Tuning
    peft_config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=NUM_VIRTUAL_TOKENS,
        prompt_tuning_init_text=config["prompt_tuning_init_text"],
        tokenizer_name_or_path=config["model_name"],
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer
