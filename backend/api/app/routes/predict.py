from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
from ..models import PredictionRequest, PredictionResponse, EmotionScore
from ..ml_service import EmotionClassifier
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class PredictionWithModelRequest(PredictionRequest):
    model: Optional[str] = Field(
        default="finetuned",
        description="finetuned | frozen | semi_frozen2 | semi_frozen4 | semi_frozen6"
    )


class CompareAttentionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)
    semi_variant: Literal["semi_frozen2", "semi_frozen4", "semi_frozen6"] = "semi_frozen4"


def _build_scores(result: dict):
    all_scores = [
        EmotionScore(emotion=EmotionClassifier.EMOTION_LABELS[i], score=result["all_probabilities"][i])
        for i in range(6)
    ]
    all_scores.sort(key=lambda x: x.score, reverse=True)
    return all_scores


def _predict_attention_for(model_key: str, text: str) -> dict:
    classifier = EmotionClassifier(model_key)
    result = classifier.predict_with_attention(text)
    all_scores = _build_scores(result)

    return {
        "model": model_key,
        "predicted_emotion": result["predicted_emotion"],
        "confidence": result["confidence"],
        "all_scores": [{"emotion": s.emotion, "score": s.score} for s in all_scores],
        "attention": {
            "tokens": result["tokens"],
            "scores": result["attention_scores"],
        },
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):
    """
    Predice la emoción (por defecto finetuned, como antes).
    """
    try:
        logger.info(f"Predicción solicitada: '{request.text[:50]}...'")

        classifier = EmotionClassifier("finetuned")
        result = classifier.predict(request.text)
        all_scores = _build_scores(result)

        logger.info(f"Predicción: {result['predicted_emotion']} ({result['confidence']:.2%})")

        return PredictionResponse(
            predicted_emotion=result["predicted_emotion"],
            confidence=result["confidence"],
            all_scores=all_scores
        )

    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/attention")
async def predict_emotion_with_attention(request: PredictionWithModelRequest):
    """
    Predice la emoción CON attention.
    Permite elegir modelo (finetuned/frozen/semi_frozenX).
    """
    try:
        model_key = (request.model or "finetuned").strip().lower()
        logger.info(f"Predicción con attention ({model_key}): '{request.text[:50]}...'")

        return _predict_attention_for(model_key, request.text)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error en predicción con attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/compare/attention")
async def compare_predict_with_attention(request: CompareAttentionRequest):
    """
    Devuelve 3 predicciones con attention:
    - frozen
    - semi (según semi_variant)
    - finetuned
    """
    try:
        text = request.text
        semi_key = request.semi_variant

        logger.info(f"COMPARE attention: frozen vs {semi_key} vs finetuned | '{text[:50]}...'")

        return {
            "frozen": _predict_attention_for("frozen", text),
            "semi": _predict_attention_for(semi_key, text),
            "finetuned": _predict_attention_for("finetuned", text),
            "semi_variant": semi_key,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error en compare attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))