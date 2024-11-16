# **Prompt Tuning for Multiple Models and Datasets**

This repository implements prompt tuning for fine-tuning causal language models like BLOOMZ, GPT-2, and LLaMA 2 on datasets such as SST-2 (sentiment classification) and NLI (Natural Language Inference). It provides a modular structure for training, evaluation, and dataset preprocessing with flexibility for extending to new models and datasets.

---

## **Table of Contents**

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Supported Models and Datasets](#supported-models-and-datasets)
- [Example Commands](#example-commands)
- [Saving and Loading Models](#saving-and-loading-models)
- [Contributing](#contributing)
- [License](#license)

---

## **Features**

- **Prompt Tuning:** Fine-tune causal language models with prompt-based learning.
- **Multiple Models:** Supports BLOOMZ, GPT-2, and LLaMA 2.
- **Dataset Agnostic:** Works with SST-2, NLI, and other datasets.
- **Flexible Evaluation:** Compare logits for token probabilities or directly decode outputs.
- **Modular Design:** Clean separation of dataset preprocessing, model training, and evaluation.

---

## **Installation**

### **Prerequisites**
- Python 3.8+
- CUDA-enabled GPU (if running on large models like LLaMA 2)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/get-started/locally/)

### **Install Dependencies**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/prompt-tuning.git
   cd prompt-tuning
