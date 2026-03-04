"""LLM chat endpoint using transformers - Production Ready."""
import modal
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uuid
from datetime import datetime

app = modal.App("llm-chat")
image = modal.Image.debian_slim().pip_install(
    "transformers", "torch", "accelerate"
)

class ChatBot:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.sessions = {}

    def setup(self):
        """Load model on container startup."""
        model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def chat(self, text: str, session_id: str = None,
             max_length: int = 100, temperature: float = 0.7) -> dict:
        """Chat with the model."""
        inputs = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store in session if session_id provided
        if session_id:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append({
                "timestamp": datetime.now().isoformat(),
                "user": text,
                "assistant": response
            })

        return {
            "user_input": text,
            "assistant_response": response,
            "session_id": session_id,
        }

chatbot = ChatBot()

@app.function(image=image, gpu="A10G", timeout=600)
@modal.enter()
def load_model():
    chatbot.setup()

@app.function(image=image, gpu="A10G", timeout=600)
@modal.web_endpoint(method="POST")
def chat(data: dict):
    """Chat with the LLM."""
    try:
        # Validate input
        text = data.get("text", "")
        if not text or not isinstance(text, str):
            return {"error": "Valid 'text' field required"}, 400

        # Sanitize input
        text = text[:2000]  # Limit length

        # Get parameters
        session_id = data.get("session_id", str(uuid.uuid4()))
        max_length = min(int(data.get("max_length", 100)), 500)
        temperature = float(data.get("temperature", 0.7))
        temperature = max(0.1, min(temperature, 2.0))  # Clamp 0.1-2.0

        # Chat
        result = chatbot.chat(text, session_id, max_length, temperature)
        return result

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}, 500

@app.function(image=image)
@modal.web_endpoint(method="GET")
def get_session(session_id: str):
    """Get chat session history."""
    if session_id in chatbot.sessions:
        return {"session_id": session_id, "history": chatbot.sessions[session_id]}
    return {"error": "Session not found"}, 404
