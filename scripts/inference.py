from transformers import AutoModelForCausalLM, GPT2TokenizerFast
import torch

# Load model and tokenizer
model_path = "../models/recipe-gpt"
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()  # set to inference mode

# Set up device agnostic code
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
# print("Using device:", device)

model.to(device)

# Define generation function
def generate_recipe(ingredients, max_length=300, temperature=0.8, top_k=50, top_p=0.95):
    prompt = "<start>\nIngredients:\n"
    for ing in ingredients:
        prompt += f"- {ing}\n"
    prompt += "Directions:\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_length=180,
            eos_token_id=tokenizer.convert_tokens_to_ids("<end>")
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Extract directions block
    if "Directions:" in generated:
        generated = generated.split("Directions:")[1]
    if "<end>" in generated:
        generated = generated.split("<end>")[0]

    return generated.strip()

def get_ingredients_from_user():
    user_input = input("Enter ingredients (comma-separated): ")
    ingredients = [i.strip() for i in user_input.split(",") if i.strip()]
    return ingredients

# Try generation
ingredients = get_ingredients_from_user()
recipe = generate_recipe(ingredients)
print("\nGenerated Recipe Directions:\n")
print(recipe)
