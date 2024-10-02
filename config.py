# config.py

# Dataset-specific configurations
DATASET_CONFIGS = {
    "sst2": {
        "name": "glue",
        "config": "sst2",
        "classes": ["negative", "positive"],
        "prompt_template": "Classify sentiment as positive or negative.\nSentence: {sentence}\nSentiment:"
    },
    "nli": {
        "name": "snli",
        "config": None,
        "classes": ["entailment", "neutral", "contradiction"],
        "prompt_template": "Determine the relationship between the sentences.\nPremise: {premise}\nHypothesis: {hypothesis}\nRelation:"
    },
}
DEFAULT_DATASET = "sst2"

# Model configurations
MODEL_CONFIGS = {
    "bloomz": {
        "model_name": "bigscience/bloomz-560m",
        "prompt_tuning_init_text": "Classify the task-specific information correctly.\n",
    },
    "gpt2": {
        "model_name": "gpt2",
        "prompt_tuning_init_text": "Answer the following task.\n",
    },
    "llama": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "prompt_tuning_init_text": "Classify the task-specific information correctly into positive or negative.\n",
    },
}
DEFAULT_MODEL = "bloomz"

# PEFT (Prompt Tuning) Configuration
NUM_VIRTUAL_TOKENS = 20

# Training settings
OUTPUT_DIR = "pt_model"
MAX_LENGTH = 64
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
