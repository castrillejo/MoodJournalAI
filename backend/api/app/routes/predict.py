from fastapi import APIRouter, HTTPException
from ..models import PredictionRequest, PredictionResponse, EmotionScore
from ..ml_service import EmotionClassifier
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict_emotion(request: PredictionRequest):
    """
    Predice la emoción de un texto en inglés
    
    **Entrada:** Texto (mín 1 carácter, máx 512)
    
    **Salida:** Emoción predicha con confianza y scores de todas las emociones
    
    **Emociones posibles:**
    - joy (alegría)
    - sadness (tristeza)
    - fear (miedo)
    - anger (ira)
    - love (amor)
    - surprise (sorpresa)
    """
    try:
        logger.info(f"Predicción solicitada para: '{request.text[:50]}...'")
        
        # Obtener clasificador (singleton)
        classifier = EmotionClassifier()
        
        # Hacer predicción
        result = classifier.predict(request.text)
        
        # Construir lista de scores para todas las emociones
        all_scores = [
            EmotionScore(
                emotion=EmotionClassifier.EMOTION_LABELS[i],
                score=result["all_probabilities"][i]
            )
            for i in range(6)
        ]
        
        # Ordenar por score descendente
        all_scores.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Predicción: {result['predicted_emotion']} ({result['confidence']:.2%})")
        
        return PredictionResponse(
            predicted_emotion=result["predicted_emotion"],
            confidence=result["confidence"],
            all_scores=all_scores
        )
        
    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Modelo no encontrado. Verifica que el modelo esté entrenado en model-training/download-model/roberta-base-english/finetuned-emotion/"
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/attention")
async def predict_emotion_with_attention(request: PredictionRequest):
    """
    Predice la emoción CON visualización de attention weights
    
    **Entrada:** Texto (mín 1 carácter, máx 512)
    
    **Salida:** Emoción predicha + tokens + attention scores
    
    Esto permite visualizar qué palabras fueron más importantes para la predicción.
    Los attention scores indican cuánta "atención" prestó el modelo a cada palabra.
    
    **Ejemplo de uso:**
    - Identificar palabras clave que determinaron la emoción
    - Explicabilidad del modelo (XAI - Explainable AI)
    - Debugging de predicciones incorrectas
    """
    try:
        logger.info(f"Predicción con attention para: '{request.text[:50]}...'")
        
        # Obtener clasificador (singleton)
        classifier = EmotionClassifier()
        
        # Hacer predicción con attention
        result = classifier.predict_with_attention(request.text)
        
        # Construir lista de scores para todas las emociones
        all_scores = [
            EmotionScore(
                emotion=EmotionClassifier.EMOTION_LABELS[i],
                score=result["all_probabilities"][i]
            )
            for i in range(6)
        ]
        
        # Ordenar por score descendente
        all_scores.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Predicción: {result['predicted_emotion']} ({result['confidence']:.2%})")
        logger.info(f"Tokens con atención: {len(result['tokens'])}")
        
        # Retornar predicción + datos de attention
        return {
            "predicted_emotion": result["predicted_emotion"],
            "confidence": result["confidence"],
            "all_scores": [{"emotion": s.emotion, "score": s.score} for s in all_scores],
            "attention": {
                "tokens": result["tokens"],
                "scores": result["attention_scores"]
            }
        }
        
    except FileNotFoundError as e:
        logger.error(f"Modelo no encontrado: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Modelo no encontrado. Verifica que el modelo esté entrenado."
        )
    except Exception as e:
        logger.error(f"Error en predicción con attention: {e}")
        raise HTTPException(status_code=500, detail=str(e))

