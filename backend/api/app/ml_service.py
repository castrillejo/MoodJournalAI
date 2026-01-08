from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionClassifier:

    EMOTION_LABELS = ["joy", "sadness", "fear", "anger", "love", "surprise"]

    _instances: Dict[str, "EmotionClassifier"] = {}

    def __new__(cls, model_key: str = "finetuned"):
        key = str(model_key).strip().lower()
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, model_key: str = "finetuned"):
        # Evitar re-cargar si ya existe en cache
        if getattr(self, "_loaded", False):
            return

        self._loaded = False
        self._model = None
        self._tokenizer = None
        self._device = None

        self.model_key = str(model_key).strip().lower()
        self.load_model(self.model_key)
        self._loaded = True

    @staticmethod
    def _project_root() -> Path:
        # backend/api/app/ml_service.py -> parents[3] = C:\MoodJournalAI
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _model_root_dir(cls) -> Path:
        root = cls._project_root()
        return root / "model-training" / "download-model" / "roberta-base-english"

    @classmethod
    def _resolve_model_dir(cls, model_key: str) -> Path:
        model_root = cls._model_root_dir()

        mapping = {
            "finetuned": "finetuned-emotion",
            "frozen": "frozen-classifier",
            "semi_frozen2": "semi-frozen2",
            "semi_frozen4": "semi-frozen4",
            "semi_frozen6": "semi-frozen6",
        }

        if model_key not in mapping:
            raise ValueError(
                f"Modelo no soportado: {model_key}. "
                f"Usa: finetuned | frozen | semi_frozen2 | semi_frozen4 | semi_frozen6"
            )

        primary = model_root / mapping[model_key]
        if primary.exists():
            return primary

        # Fallback mínimo por si tus carpetas usan '_' en vez de '-'
        alt = model_root / mapping[model_key].replace("-", "_")
        if alt.exists():
            return alt

        raise FileNotFoundError(f"Modelo no encontrado en: {primary} (ni en {alt})")

    def load_model(self, model_key: str):
        """Carga el modelo desde disco según model_key (tokenizer incluido en la carpeta del modelo)."""
        model_path = self._resolve_model_dir(model_key)

        logger.info(f"Cargando modelo '{model_key}' desde: {model_path}")

        # Cargar tokenizer y modelo (desde la MISMA carpeta del modelo)
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

        # GPU si hay
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

        logger.info(f"✅ Modelo '{model_key}' cargado en {self._device.upper()}")

    def predict(self, text: str) -> Dict:
        """Predicción sin attention."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)

        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        all_probs = probabilities[0].detach().cpu().numpy().tolist()

        return {
            "predicted_class": predicted_class_id,
            "predicted_emotion": self.EMOTION_LABELS[predicted_class_id],
            "confidence": confidence,
            "all_probabilities": all_probs,
        }

    def predict_with_attention(self, text: str) -> Dict:
        """Predicción con attention (como tu original)."""
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions
            probabilities = torch.softmax(logits, dim=-1)

        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        all_probs = probabilities[0].detach().cpu().numpy().tolist()

        # Última capa: (batch, heads, seq, seq)
        last_layer_attention = attentions[-1][0]  # primer batch
        avg_attention = last_layer_attention.mean(dim=0)  # (seq, seq)
        cls_attention = avg_attention[0].detach().cpu().numpy()  # atención del token 0 al resto

        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        clean_tokens: List[str] = []
        attention_scores: List[float] = []

        for token, score in zip(tokens, cls_attention):
            if token in ["<s>", "</s>", "<pad>"]:
                continue

            token_clean = token.replace("Ġ", "")
            if token_clean.strip() == "":
                continue

            clean_tokens.append(token_clean)
            attention_scores.append(float(score))

        # Normalizar 0..1
        if attention_scores:
            min_score = min(attention_scores)
            max_score = max(attention_scores)
            if max_score - min_score > 1e-12:
                attention_scores = [
                    (s - min_score) / (max_score - min_score) for s in attention_scores
                ]
            else:
                attention_scores = [0.0 for _ in attention_scores]

        return {
            "predicted_class": predicted_class_id,
            "predicted_emotion": self.EMOTION_LABELS[predicted_class_id],
            "confidence": confidence,
            "all_probabilities": all_probs,
            "tokens": clean_tokens,
            "attention_scores": attention_scores,
        }
