from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request para predicción de emoción"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=512,
        description="Texto en inglés para clasificar",
        examples=["I feel so happy today!"]
    )

class EmotionScore(BaseModel):
    """Score de una emoción individual"""
    emotion: str = Field(..., description="Nombre de la emoción")
    score: float = Field(..., ge=0.0, le=1.0, description="Probabilidad (0-1)")

class PredictionResponse(BaseModel):
    """Response con predicción de emoción"""
    predicted_emotion: str = Field(..., description="Emoción predicha")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la predicción")
    all_scores: List[EmotionScore] = Field(..., description="Scores de todas las emociones")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "predicted_emotion": "joy",
                    "confidence": 0.94,
                    "all_scores": [
                        {"emotion": "joy", "score": 0.94},
                        {"emotion": "love", "score": 0.03},
                        {"emotion": "surprise", "score": 0.02}
                    ]
                }
            ]
        }
    }
