from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionClassifier:
    """
    Singleton para gestionar el modelo de clasificación de emociones.
    Se carga una sola vez al iniciar la aplicación.
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    # Mapeo de IDs a nombres de emociones
    EMOTION_LABELS = {
        0: "joy",
        1: "sadness",
        2: "fear",
        3: "anger",
        4: "love",
        5: "surprise"
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self.load_model()
    
    def load_model(self):
        """Carga el modelo fine-tuned desde disco"""
        # Ruta al modelo - ajustada a tu estructura real
        # backend/api/app -> ../../.. = root -> model-training/download-model/roberta-base-english/finetuned-emotion
        base_path = Path(__file__).parent.parent.parent.parent
        model_path = base_path / "model-training" / "download-model" / "roberta-base-english" / "finetuned-emotion"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        logger.info(f"Cargando modelo desde: {model_path}")
        
        # Cargar tokenizer y modelo
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Detectar y usar GPU si está disponible
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()  # Modo evaluación (no entrenamiento)
        
        logger.info(f"✅ Modelo cargado en {self._device.upper()}")
    
    def predict(self, text: str) -> Dict:
        """
        Predice la emoción de un texto
        
        Args:
            text: Texto en inglés para clasificar
            
        Returns:
            Dict con predicted_class, confidence y all_probabilities
        """
        # Tokenizar
        inputs = self._tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=True
        )
        
        # Mover a GPU si disponible
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Hacer predicción (sin calcular gradientes)
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Obtener clase predicha y confianza
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        
        # Todas las probabilidades (convertir a lista Python)
        all_probs = probabilities[0].cpu().numpy().tolist()
        
        return {
            "predicted_class": predicted_class_id,
            "predicted_emotion": self.EMOTION_LABELS[predicted_class_id],
            "confidence": confidence,
            "all_probabilities": all_probs
        }
    
    def predict_with_attention(self, text: str) -> Dict:
        """
        Predice la emoción y extrae attention weights para visualización
        
        Args:
            text: Texto en inglés para clasificar
            
        Returns:
            Dict con predicción + tokens + attention scores
        """
        # Tokenizar
        inputs = self._tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding=True,
            return_offsets_mapping=False
        )
        
        # Mover a GPU si disponible
        inputs_device = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Hacer predicción CON attention weights
        with torch.no_grad():
            outputs = self._model(**inputs_device, output_attentions=True)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            attentions = outputs.attentions  # Tuple de (12 capas, batch, heads, seq_len, seq_len)
        
        # Obtener predicción
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_class_id].item()
        all_probs = probabilities[0].cpu().numpy().tolist()
        
        # Extraer attention de la última capa
        # attentions[-1] = última capa transformer
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        last_layer_attention = attentions[-1][0]  # Primer batch
        
        # Promediar sobre todos los heads
        # Shape: (seq_len, seq_len)
        avg_attention = last_layer_attention.mean(dim=0)
        
        # Attention del token [CLS] (primera posición) a todos los demás tokens
        # Esto indica qué tan importante fue cada token para la clasificación
        cls_attention = avg_attention[0].cpu().numpy()
        
        # Convertir input_ids a tokens legibles
        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Limpiar tokens especiales de RoBERTa (Ġ indica espacio)
        clean_tokens = []
        attention_scores = []
        
        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            # Saltar tokens especiales al inicio/final
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            # Limpiar prefijo Ġ (indica palabra que empieza después de espacio)
            clean_token = token.replace('Ġ', ' ')
            clean_tokens.append(clean_token)
            attention_scores.append(float(score))
        
        # Normalizar scores para mejor visualización (0-1)
        if attention_scores:
            min_score = min(attention_scores)
            max_score = max(attention_scores)
            if max_score > min_score:
                attention_scores = [
                    (score - min_score) / (max_score - min_score) 
                    for score in attention_scores
                ]
        
        return {
            "predicted_class": predicted_class_id,
            "predicted_emotion": self.EMOTION_LABELS[predicted_class_id],
            "confidence": confidence,
            "all_probabilities": all_probs,
            "tokens": clean_tokens,
            "attention_scores": attention_scores
        }

