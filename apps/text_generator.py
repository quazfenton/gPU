"""Text generation with GPT-2 - Production Ready."""
import modal
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

app = modal.App("text-generator")
image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "accelerate"
)

class TextGenerator:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def setup(self):
        """Load model on container startup."""
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

    def generate(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

generator = TextGenerator()

@app.function(image=image, gpu="T4", timeout=600)
@modal.enter()
def load_model():
    generator.setup()

@app.function(image=image, gpu="T4", timeout=600)
@modal.web_endpoint(method="POST")
def generate(data: dict):
    """Generate text from prompt."""
    try:
        # Validate input
        if not data or "prompt" not in data:
            return {"error": "Missing 'prompt' field"}, 400

        prompt = str(data.get("prompt", ""))[:2000]  # Limit input length

        # Sanitize input (remove potentially harmful content)
        prompt = re.sub(r'<[^>]*>', '', prompt)  # Remove HTML tags

        # Validate max_length
        max_length = int(data.get("max_length", 100))
        max_length = min(max(50, max_length), 1024)  # Clamp to 50-1024

        # Generate
        full_text = generator.generate(prompt, max_length)
        generated_text = full_text[len(prompt):].strip()

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "total_length": len(full_text),
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}, 500
