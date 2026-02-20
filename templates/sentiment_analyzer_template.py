"""
Sentiment Analyzer — analyze text sentiment using transformers.

Automates the popular Colab workflow:
  1. Enter text or upload document
  2. Run sentiment analysis model
  3. Get back sentiment labels and scores
"""

from typing import Any, Dict

from templates.base import Template, InputField, OutputField, RouteType


class SentimentAnalyzerTemplate(Template):
    name = "sentiment-analyzer"
    category = "Text"
    description = (
        "Analyze text sentiment using state-of-the-art transformers. "
        "Returns sentiment label (positive, negative, neutral), confidence "
        "scores, and emotion analysis. Supports multiple languages."
    )
    version = "1.0.0"

    inputs = [
        InputField(name="text", type="text", description="Text to analyze", required=True),
        InputField(
            name="model",
            type="text",
            description="Sentiment model",
            required=False,
            default="distilbert-base-uncased-finetuned-sst-2-english",
        ),
        InputField(
            name="analyze_emotions",
            type="text",
            description="Also analyze specific emotions",
            required=False,
            default="false",
            options=["true", "false"],
        ),
    ]

    outputs = [
        OutputField(name="sentiment", type="text", description="Sentiment label (POSITIVE/NEGATIVE/NEUTRAL)"),
        OutputField(name="score", type="number", description="Confidence score (0-1)"),
        OutputField(name="emotions", type="json", description="Emotion analysis if requested"),
    ]

    routing = [RouteType.LOCAL, RouteType.MODAL, RouteType.HF]
    gpu_required = False
    gpu_type = None
    memory_mb = 2048
    timeout_sec = 60
    pip_packages = ["transformers", "torch"]

    def setup(self):
        self._sentiment_pipeline = None
        self._emotion_pipeline = None
        self._initialized = True

    def run(self, **kwargs) -> Dict[str, Any]:
        self.validate_inputs(**kwargs)
        from transformers import pipeline

        text = kwargs["text"]
        model = kwargs.get("model", "distilbert-base-uncased-finetuned-sst-2-english")
        analyze_emotions = kwargs.get("analyze_emotions", "false") == "true"

        if self._sentiment_pipeline is None:
            self._sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
            )

        result = self._sentiment_pipeline(text)[0]

        response = {
            "sentiment": result["label"].upper(),
            "score": round(result["score"], 4),
        }

        if analyze_emotions:
            if self._emotion_pipeline is None:
                self._emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                )
            emotions = self._emotion_pipeline(text)[0]
            response["emotions"] = [
                {"label": e["label"], "score": round(e["score"], 4)}
                for e in emotions
            ]

        return response
