import gradio as gr
import torch
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

# Load fine-tuned model
model_path = "saksh-d/recipe-gpt"
tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Device setup
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def generate(ingredient_text, temperature, top_k, top_p, max_length):
    # Format ingredients into a list
    ingredients = [line.strip("- ").strip() for line in ingredient_text.strip().splitlines() if line.strip()]
    
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
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_length=max_length,
            eos_token_id=tokenizer.convert_tokens_to_ids("<end>")
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    if "Directions:" in generated:
        generated = generated.split("Directions:")[1]
    if "<end>" in generated:
        generated = generated.split("<end>")[0]

    return generated.strip()

iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(lines=8, label="Ingredients (one per line)"),
        gr.Slider(minimum=0.5, maximum=1.5, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0, maximum=100, value=40, step=5, label="Top-k"),
        gr.Slider(minimum=0.5, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
        gr.Slider(minimum=50, maximum=150, value=120, step=10, label="Recipe Length"),
    ],
    outputs=gr.Textbox(lines=12, label="Generated Recipe Directions"),
    title="Recipe-GPT",
    description="Enter a list of ingredients to generate step-by-step cooking directions. Adjust the sliders for more or less creativity."
)

iface.launch()
