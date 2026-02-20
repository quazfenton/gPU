"""
Translator — translate text between languages using NLLB or Helsinki-NLP.

Automates the popular Colab workflow:
  1. Enter text
  2. Select source and target language
  3. Get back translated text
"""

from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class TranslatorTemplate(Template):
    name = "translator"
    category = "Text"
    description = (
        "Translate text between 200+ languages using Meta's NLLB model. "
        "Supports low-resource languages and provides high-quality translations. "
        "Great for multilingual content, localization, and communication."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="text", type="text", description="Text to translate", required=True),
        InputField(
            name="source_language",
            type="text",
            description="Source language code ('auto' for detection)",
            required=False,
            default="auto",
        ),
        InputField(
            name="target_language",
            type="text",
            description="Target language code",
            required=True,
            default="en",
        ),
        InputField(
            name="model",
            type="text",
            description="Translation model",
            required=False,
            default="facebook/nllb-200-distilled-600M",
        ),
    ]

    outputs = [
        OutputField(name="translated_text", type="text", description="Translated text"),
        OutputField(name="detected_language", type="text", description="Detected source language"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 4096
    timeout_sec = 180
    pip_packages = ["transformers", "torch", "sentencepiece"]

    def setup(self):
        self._pipeline = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from transformers import pipeline

        text = kwargs["text"]
        source = kwargs.get("source_language", "auto")
        target = kwargs.get("target_language", "en")
        model = kwargs.get("model", "facebook/nllb-200-distilled-600M")

        if self._pipeline is None:
            self._pipeline = pipeline(
                "translation",
                model=model,
                torch_dtype=__import__("torch").float16,
            )

        src_lang = source if source != "auto" else None
        result = self._pipeline(text, src_lang=src_lang, tgt_lang=target)

        detected = source if source != "auto" else "auto"

        return {
            "translated_text": result[0]["translation_text"],
            "detected_language": detected,
        }
