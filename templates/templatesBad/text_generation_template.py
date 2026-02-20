"""
Text Generation Template for generating text using large language models.

This template uses transformer-based language models (e.g., GPT-2) to generate
coherent text continuations from a given prompt, supporting various sampling
parameters for controlling output quality and creativity.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class TextGenerationTemplate(Template):
    """
    Generates text from a prompt using language models.
    
    Uses pre-trained transformer models (e.g., GPT-2) to generate text
    continuations. Supports controlling generation via temperature, top-p
    sampling, and maximum length parameters.
    
    Returns generated text and token usage statistics.
    """
    
    name = "text-generation"
    category = "Language"
    description = "Generate text continuations from a prompt using language models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="prompt",
            type="text",
            description="Text prompt to generate from",
            required=True
        ),
        InputField(
            name="max_length",
            type="number",
            description="Maximum number of tokens to generate",
            required=False,
            default=200
        ),
        InputField(
            name="temperature",
            type="number",
            description="Sampling temperature (higher = more creative)",
            required=False,
            default=0.7
        ),
        InputField(
            name="top_p",
            type="number",
            description="Nucleus sampling probability threshold",
            required=False,
            default=0.9
        )
    ]
    
    outputs = [
        OutputField(
            name="generated_text",
            type="text",
            description="Generated text continuation"
        ),
        OutputField(
            name="tokens_used",
            type="number",
            description="Total number of tokens in the generated output"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 8192
    timeout_sec = 300
    pip_packages = ["transformers", "torch", "accelerate"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the text generation model.
        
        Loads a pre-trained GPT-2 model and tokenizer from HuggingFace.
        The model is moved to GPU if available.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token to eos token for open-ended generation
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute text generation on the provided prompt.
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum number of tokens to generate (optional, defaults to 200)
            temperature: Sampling temperature (optional, defaults to 0.7)
            top_p: Nucleus sampling threshold (optional, defaults to 0.9)
            
        Returns:
            Dict containing:
                - generated_text: The generated text continuation
                - tokens_used: Number of tokens in the generated output
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import torch
        
        # Extract parameters
        prompt = kwargs['prompt']
        max_length = int(kwargs.get('max_length', 200))
        temperature = float(kwargs.get('temperature', 0.7))
        top_p = float(kwargs.get('top_p', 0.9))
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        
        # Calculate tokens used (generated only, excluding prompt)
        tokens_used = outputs.shape[1] - input_length
        
        return {
            'generated_text': generated_text,
            'tokens_used': int(tokens_used)
        }
