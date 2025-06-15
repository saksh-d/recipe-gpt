from transformers import GPT2TokenizerFast
NUMBER_OF_RECIPE_BATCHES = 50000

def create_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    special_tokens = {
    "bos_token" : "<start>",
    "eos_token" : "<end>",
    "additional_special_tokens": []
    }

    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def process_recipes(recipes, tokenizer):
    """
    Splits recipes by "<start>", prepends "<start>" back to each block,
    and tokenizes each block.
    
    Args:
        recipes (str): Raw recipe text containing multiple recipes separated by "<start>"
        tokenizer (callable): Tokenizer function to apply to each recipe block
    
    Returns:
        list: List of tokenized recipe blocks
    """
    
    # Split by "<start>" and remove empty strings
    recipe_blocks = [block.strip() for block in recipes.split("<start>") if block.strip()]

    # Prepend <start> back to each block and tokenize
    tokenized_recipes = []
    for i, block in enumerate(recipe_blocks):
        if i == NUMBER_OF_RECIPE_BATCHES:  # Processing recipe blocks due to memory constraints
            break
        full_block = "<start>\n" + block
        tokenized_block = tokenizer(full_block, truncation=True, max_length=512, padding=False)
        tokenized_recipes.append(tokenized_block)
    
    return tokenized_recipes