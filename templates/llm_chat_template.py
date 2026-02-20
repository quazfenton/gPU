"""
LLM Chat — chat with large language models.

Automates workflows for:
  1. Text generation with LLMs
  2. Instruction-following
  3. Code generation
  4. Conversation with context
"""

import os
from typing import Any, Dict, List

from templates.base import Template, InputField, OutputField, RouteType


class LLMChatTemplate(Template):
    name = "llm-chat"
    category = "Text"
    description = (
        "Chat with state-of-the-art large language models. Generate text, answer questions, "
        "write code, summarize documents, and more. Supports GPT-4, Claude, Llama, and other "
        "popular LLMs. Perfect for building chatbots, content generation, and reasoning tasks."
    )
    version = "1.0.0"

    inputs = [
        InputField(
            name="prompt",
            type="text",
            description="User message or task description",
            required=True,
        ),
        InputField(
            name="model",
            type="text",
            description="LLM to use",
            required=False,
            default="gpt-3.5-turbo",
            options=[
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "claude-3-opus",
                "claude-3-sonnet",
                "claude-3-haiku",
                "llama-3-70b",
                "llama-3-8b",
                "mistral-7b",
                "mixtral-8x7b",
            ],
        ),
        InputField(
            name="system_prompt",
            type="text",
            description="System instructions",
            required=False,
            default="You are a helpful AI assistant.",
        ),
        InputField(
            name="temperature",
            type="number",
            description="Creativity (0-2, higher = more creative)",
            required=False,
            default=0.7,
        ),
        InputField(
            name="max_tokens",
            type="number",
            description="Maximum response length",
            required=False,
            default=2048,
        ),
        InputField(
            name="top_p",
            type="number",
            description="Nucleus sampling (0-1)",
            required=False,
            default=1.0,
        ),
        InputField(
            name="presence_penalty",
            type="number",
            description="Presence penalty (-2 to 2)",
            required=False,
            default=0.0,
        ),
        InputField(
            name="frequency_penalty",
            type="number",
            description="Frequency penalty (-2 to 2)",
            required=False,
            default=0.0,
        ),
        InputField(
            name="stop_sequences",
            type="json",
            description="Stop sequences",
            required=False,
        ),
    ]

    outputs = [
        OutputField(name="response", type="text", description="Model response")),
        OutputField(name="usage", type="json", description="Token usage statistics")),
        OutputField(name="model_used", type="text", description="Model that generated response")),
        OutputField(name="finish_reason", type="text", description="Why generation stopped")),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 180
    pip_packages = ["openai", "anthropic", "transformers", "torch"]

    def setup(self):
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)

        user_prompt = kwargs["prompt"]
        model = kwargs.get("model", "gpt-3.5-turbo")
        system = kwargs.get("system_prompt", "You are a helpful AI assistant.")
        temp = float(kwargs.get("temperature", 0.7))
        max_tokens = int(kwargs.get("max_tokens", 2048))
        top_p = float(kwargs.get("top_p", 1.0))
        presence = float(kwargs.get("presence_penalty", 0.0))
        frequency = float(kwargs.get("frequency_penalty", 0.0))
        stop = kwargs.get("stop_sequences")

        if model.startswith("gpt"):
            try:
                import openai
                client = openai.OpenAI()

                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ]

                result = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence,
                    frequency_penalty=frequency,
                    stop=stop,
                )

                response = {
                    "response": result.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                    },
                    "model_used": result.model,
                    "finish_reason": result.choices[0].finish_reason,
                }

            except Exception as e:
                response = {
                    "response": f"Error: {str(e)}",
                    "usage": {},
                    "model_used": model,
                    "finish_reason": "error",
                }

        elif model.startswith("claude"):
            try:
                import anthropic
                client = anthropic.Anthropic()

                result = client.messages.create(
                    model="claude-3-opus-20240229" if "opus" in model else
                          "claude-3-sonnet-20240229" if "sonnet" in model else
                          "claude-3-haiku-20240229",
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temp,
                    top_p=top_p,
                )

                response = {
                    "response": result.content[0].text,
                    "usage": {
                        "input_tokens": result.usage.input_tokens,
                        "output_tokens": result.usage.output_tokens,
                    },
                    "model_used": model,
                    "finish_reason": "stop",
                }

            except Exception as e:
                response = {
                    "response": f"Error: {str(e)}",
                    "usage": {},
                    "model_used": model,
                    "finish_reason": "error",
                }

        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            full_prompt = f"System: {system}\n\nUser: {user_prompt}\n\nAssistant:"

            inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=top_p,
                    do_sample=temp > 0,
                )

            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = response_text[len(full_prompt):].strip()

            response = {
                "response": response_text,
                "usage": {"tokens_generated": outputs.shape[1]},
                "model_used": model,
                "finish_reason": "stop",
            }

        return response
