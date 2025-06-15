import os
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from utils import process_recipes, create_tokenizer

dataset = 'tengomucho/all-recipes-split'
dataset_split = 'train'
tokenizer = create_tokenizer()


def get_dataset(dataset, dataset_split):
    """
    Function to grab dataset based on `dataset` and `dataset_split`.
    """
    data = load_dataset(dataset, split=dataset_split)
    
    return data

def save_data(dataset, dataset_split):
    """
    Function to save data locally as a text file.
    """
    data = get_dataset(dataset, dataset_split)
    data_path = Path('../data/')
    file_path = data_path / "recipes.txt"

    if file_path.exists():
        print("Dataset file already exists. Moving onto tokenization...")
        return

    if data_path.is_dir():
        print(f"Directory exists, skipping...")
    else:
        print("Creating diretory...")
        data_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "recipes.txt", "w", encoding="utf-8") as f:
        for i in range(len(data)):
            title = data[i].get("title", "").strip()
            ingredients = data[i].get("ingredients", "").strip()
            directions = data[i].get("directions", "").strip()

            recipe_block = f"<start>\n"  # Add special token to the beginning of each recipe
            if title:
                recipe_block += f"Title: {title}\n"
            recipe_block += f"Ingredients:\n{ingredients}\n"
            recipe_block += f"Directions:\n{directions}\n"
            recipe_block += f"<end>\n\n"  # Add a special token to the end of each recipe

            f.write(recipe_block) 

    print("File saved.")


def create_dataset(processed_recipes):
    """
    Extract input_ids and attention_mask from processed recipes and create a Dataset.
    
    Args:
        processed_recipes (list): List of tokenized recipe blocks
    
    Returns:
        Dataset: HuggingFace Dataset with input_ids and attention_mask
    """
    input_ids = []
    attention_mask = []

    for recipe in processed_recipes:
        input_ids.append(recipe['input_ids'])
        attention_mask.append(recipe['attention_mask'])

    dataset = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })

    return dataset


def split_and_save_dataset(dataset, train_split=0.9, save_dir="../data/tokenized_recipes"):
    """
    Split tokenized dataset into train/validation and save both to disk.
    
    Args:
        dataset (Dataset): HuggingFace Dataset to split
        train_split (float): Proportion for training set (default 0.9 = 90%)
        save_dir (str): Directory to save the datasets
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """

    # Split the dataset
    split_dataset = dataset.train_test_split(test_size=1-train_split, seed=42)
    train_dataset = split_dataset['train']
    val_dataset = split_dataset['test']
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save datasets to disk
    train_path = os.path.join(save_dir, "train")
    val_path = os.path.join(save_dir, "validation")
    
    train_dataset.save_to_disk(train_path)
    val_dataset.save_to_disk(val_path)
    
    print(f"Datasets saved successfully!")
    print(f"Train dataset: {train_path} | ({len(train_dataset)} samples)")
    print(f"Validation dataset: {val_path} | ({len(val_dataset)} samples)")
    
    return train_dataset, val_dataset


def load_saved_datasets(save_dir="../data/tokenized_recipes"):
    """
    Load previously saved tokenized train/validation datasets from disk.
    
    Args:
        save_dir (str): Directory where datasets were saved
    
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    
    train_path = os.path.join(save_dir, "train")
    val_path = os.path.join(save_dir, "validation")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Datasets not found in {save_dir}. Please run processing first.")
    
    train_dataset = Dataset.load_from_disk(train_path)
    val_dataset = Dataset.load_from_disk(val_path)
    
    print(f"Datasets loaded successfully!")
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    #download data
    get_dataset(dataset, dataset_split)
    #save as text file
    save_data(dataset, dataset_split)

    #Create tokenized dataset
    with open("../data/recipes.txt", "r", encoding="utf-8") as f:
        recipes = f.read()

    processed_recipes = process_recipes(recipes, tokenizer)
    dataset = create_dataset(processed_recipes)
    split_and_save_dataset(dataset)
    print("Successfully created tokenized dataset.")
    