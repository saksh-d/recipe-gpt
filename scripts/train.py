import torch
from utils import create_tokenizer
from dataset import load_saved_datasets
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# HYPERPARAMETERS
save_dir = "..data/tokenized_recipes"
NUM_EPOCHS = 5

# Set up device agnostic code
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print("Using device:", device)

tokenizer = create_tokenizer()

# Load dataset
train_dataset, val_dataset = load_saved_datasets(save_dir)

# Set up model
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm = False #Causal LM, not BERT
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="../models/recipe-gpt",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    save_total_limit=2,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=4,  # adjust based on available memory
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    fp16=True,  # Set to True if using a GPU with FP16 support
    report_to="none"
)

#Set up trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()
final_model_path = "../models/recipe-gpt"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
