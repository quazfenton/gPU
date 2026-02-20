"""
Code Generation Template for generating code from natural language prompts.

This template uses transformer-based code models to generate code snippets
from natural language descriptions, supporting multiple programming languages
with configurable generation parameters.
"""

from typing import Any, Dict
from templates.base import Template, InputField, OutputField, RouteType


class CodeGenerationTemplate(Template):
    """
    Generates code from natural language descriptions.
    
    Uses pre-trained code generation models (e.g., Salesforce CodeGen) to
    produce code from text prompts. Supports multiple programming languages
    and configurable generation parameters for controlling output quality.
    
    Returns generated code and the detected language.
    """
    
    name = "code-generation"
    category = "Language"
    description = "Generate code from natural language prompts using code models"
    version = "1.0.0"
    
    inputs = [
        InputField(
            name="prompt",
            type="text",
            description="Natural language description of the code to generate",
            required=True
        ),
        InputField(
            name="language",
            type="text",
            description="Target programming language",
            required=False,
            default="python",
            options=["python", "javascript", "java", "cpp", "rust"]
        ),
        InputField(
            name="max_length",
            type="number",
            description="Maximum number of tokens to generate",
            required=False,
            default=256
        ),
        InputField(
            name="temperature",
            type="number",
            description="Sampling temperature (lower = more deterministic)",
            required=False,
            default=0.2
        )
    ]
    
    outputs = [
        OutputField(
            name="code",
            type="text",
            description="Generated code snippet"
        ),
        OutputField(
            name="language",
            type="text",
            description="Programming language of the generated code"
        )
    ]
    
    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = True
    gpu_type = "T4"
    memory_mb = 4096
    timeout_sec = 120
    pip_packages = ["transformers", "torch"]
    
    def setup(self) -> None:
        """
        One-time initialization to load the code generation model.
        
        Loads a pre-trained CodeGen model and tokenizer from HuggingFace.
        Default model is Salesforce/codegen-350M-mono.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        model_name = "Salesforce/codegen-350M-mono"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        self._initialized = True
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute code generation from the provided prompt.
        
        Args:
            prompt: Natural language description of the code to generate
            language: Target programming language (optional, defaults to 'python')
            max_length: Maximum tokens to generate (optional, defaults to 256)
            temperature: Sampling temperature (optional, defaults to 0.2)
            
        Returns:
            Dict containing:
                - code: Generated code snippet
                - language: Programming language of the output
        """
        # Validate inputs
        self.validate_inputs(**kwargs)
        
        # Initialize if needed
        if not self._initialized:
            self.setup()
        
        import torch
        
        # Extract parameters
        prompt = kwargs['prompt']
        language = kwargs.get('language', 'python')
        max_length = int(kwargs.get('max_length', 256))
        temperature = float(kwargs.get('temperature', 0.2))
        
        # Format prompt with language context
        formatted_prompt = f"# {language}\n# {prompt}\n"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate code
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated tokens (exclude the input prompt tokens)
        generated_tokens = outputs[0][input_length:]
        code = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'code': code.strip(),
            'language': language
        }
