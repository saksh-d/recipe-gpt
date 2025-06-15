# Recipe-GPT

This project fine-tunes the pretrained [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) model using Hugging Face’s `transformers` and `datasets` libraries on a large corpus of human-written recipes.

The goal is to teach GPT-2 how to generate coherent cooking instructions from a list of ingredients.

## Key Features:
- Base Model: GPT-2 (117M) from Hugging Face Model Hub
- Frameworks: 
  - `transformers` for model loading, training, and generation
  - `datasets` for loading and managing large recipe data
  - `Trainer` API for end-to-end fine-tuning
- Custom Tokens: 
  - Added `<start>` and `<end>` tokens to mark recipe boundaries
  - Tokenizer and model embeddings resized accordingly
- Data Format:
  - Recipes formatted as plain text blocks with titles, ingredients, and step-by-step directions
- Training Strategy:
  - Causal language modeling (not masked)
  - Evaluated on validation set each epoch
  - Supports CPU, CUDA, and Apple MPS backends

This setup allows the model to learn full-text generation patterns and structure, making it effective at translating structured ingredient lists into complete, human-readable cooking instructions.

## Dataset

Source: [`tengomucho/all-recipes-split`](https://huggingface.co/datasets/tengomucho/all-recipes-split)

2.1M+ recipes with:
  - `title`
  - `ingredients`
  - `directions`

---

## How to Use

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download dataset and tokenize
```bash
python scripts/dataset.py
```

### Train the model
```bash
python scripts/train.py
```

### Run Inference
```bash
python scripts/inference.py
```

Enter comma-separated ingredients when prompted, for example,
```bash
Enter ingredients (comma-separated): chicken, rice, garlic
```
