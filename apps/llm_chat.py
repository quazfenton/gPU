"""LLM chat endpoint using transformers."""
import modal

app = modal.App("llm-chat")
image = modal.Image.debian_slim().pip_install("transformers", "torch", "accelerate")

@app.function(image=image, gpu="A10G", timeout=600)
@modal.web_endpoint(method="POST")
def chat(data: dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}, 400
    
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    inputs = tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    outputs = model.generate(
        **inputs,
        max_length=data.get("max_length", 100),
        temperature=data.get("temperature", 0.7),
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"text": text, "response": response}
