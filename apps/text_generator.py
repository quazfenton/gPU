"""Text generation with GPT-2."""
import modal

app = modal.App("text-generator")
image = modal.Image.debian_slim().pip_install("transformers", "torch")

@app.function(image=image, gpu="T4", timeout=600)
@modal.web_endpoint(method="POST")
def generate(data: dict):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    inputs = tokenizer(data.get('prompt', ''), return_tensors='pt')
    outputs = model.generate(**inputs, max_length=data.get('max_length', 100))
    
    return {"text": tokenizer.decode(outputs[0])}
