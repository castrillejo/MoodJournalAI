# Plan de Implementación: Backend API con FastAPI

## 📋 Estructura Mejorada del Proyecto

```
MoodJournalAI/
├── .venv/                        # Entorno virtual único
├── requirements.txt              # Todas las dependencias
│
├── backend/
│   ├── api/                      # 🎯 FASE 1: FastAPI (API REST)
│   │   ├── app/
│   │   │   ├── __init__.py
│   │   │   ├── main.py           # Aplicación FastAPI
│   │   │   ├── models.py         # Modelos Pydantic (request/response)
│   │   │   ├── ml_service.py     # Servicio de ML (carga modelo)
│   │   │   └── routes/
│   │   │       ├── __init__.py
│   │   │       └── predict.py    # Endpoint de predicción
│   │   └── README.md
│   │
│   └── web/                      # 🔮 FASE 2: Streamlit (Web App)
│       └── streamlit_app.py
│
├── frontend/                     # 🔮 FASE 3: React (Frontend)
│   └── (futuro)
│
├── models/                       # Modelos entrenados
│   └── final/
│       ├── config.json
│       └── model.safetensors
│
├── fine-tuning/                  # Scripts de entrenamiento
│   ├── prepare_dataset.py
│   ├── train.py
│   └── evaluate.py
│
└── data/
    └── entradas.csv
```

---

## 🎯 FASE 1: Implementación de FastAPI

### Objetivo
Crear API REST que sirva predicciones del modelo RoBERTa fine-tuned

### Arquitectura
```
Cliente → POST /api/predict → FastAPI → RoBERTa → Response JSON
```

---

## 📦 Paso 1: Dependencias

### Actualizar `requirements.txt` (raíz del proyecto)

```txt
# Machine Learning
torch==2.1.0
transformers==4.35.0
pandas==2.1.0
scikit-learn==1.3.0
datasets==2.14.0

# FastAPI
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Utils
python-multipart==0.0.6
```

### Instalar
```bash
# Activar entorno virtual
.\.venv\Scripts\Activate

# Instalar nuevas dependencias
pip install fastapi uvicorn[standard]
```

---

## 📂 Paso 2: Crear Estructura de Carpetas

```bash
# Desde la raíz del proyecto
cd C:\MoodJournalAI

# Crear estructura backend/api
mkdir backend\api\app\routes

# Crear archivos vacíos
New-Item backend\api\app\__init__.py
New-Item backend\api\app\main.py
New-Item backend\api\app\models.py
New-Item backend\api\app\ml_service.py
New-Item backend\api\app\routes\__init__.py
New-Item backend\api\app\routes\predict.py
New-Item backend\api\README.md
```

---

## 🔧 Paso 3: Implementar Componentes

### 3.1 - `backend/api/app/models.py`

**Propósito:** Definir estructura de requests y responses con Pydantic

```python
from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    """Request para predicción de emoción"""
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=512,
        description="Texto en inglés para clasificar",
        example="I feel so happy today!"
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_emotion": "joy",
                "confidence": 0.94,
                "all_scores": [
                    {"emotion": "joy", "score": 0.94},
                    {"emotion": "love", "score": 0.03},
                    {"emotion": "surprise", "score": 0.02}
                ]
            }
        }
```

**¿Qué hace?**
- `PredictionRequest`: Valida que el texto tenga 1-512 caracteres
- `EmotionScore`: Representa el score de una emoción (0.0-1.0)
- `PredictionResponse`: Estructura de la respuesta JSON

---

### 3.2 - `backend/api/app/ml_service.py`

**Propósito:** Cargar y gestionar el modelo RoBERTa (Singleton pattern)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pathlib import Path
from typing import Dict, List
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
        # Ruta al modelo (ajusta según tu estructura)
        model_path = Path(__file__).parent.parent.parent.parent / "models" / "final"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        logger.info(f"Cargando modelo desde: {model_path}")
        
        # Cargar tokenizer y modelo
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
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
```

**¿Qué hace?**
- **Singleton**: Solo carga el modelo UNA vez (ahorra memoria y tiempo)
- **Auto-detecta GPU**: Usa CUDA si está disponible
- **predict()**: Tokeniza texto → Modelo → Probabilidades

---

### 3.3 - `backend/api/app/routes/predict.py`

**Propósito:** Endpoint POST /predict

```python
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
            detail="Modelo no encontrado. Verifica que el modelo esté en models/final/"
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**¿Qué hace?**
- Valida request con Pydantic
- Llama al clasificador
- Retorna JSON con emoción predicha + scores

---

### 3.4 - `backend/api/app/main.py`

**Propósito:** Aplicación FastAPI principal

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import predict
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Crear aplicación FastAPI
app = FastAPI(
    title="MoodJournalAI API",
    description="API para clasificación de emociones usando RoBERTa fine-tuned",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS (para permitir requests desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(
    predict.router, 
    prefix="/api", 
    tags=["Predictions"]
)

@app.get("/")
async def root():
    """Endpoint raíz - información básica de la API"""
    return {
        "message": "MoodJournalAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MoodJournalAI API"
    }

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la aplicación"""
    logging.info("🚀 Iniciando MoodJournalAI API...")
    # El modelo se carga al hacer la primera predicción (lazy loading)

@app.on_event("shutdown")
async def shutdown_event():
    """Se ejecuta al cerrar la aplicación"""
    logging.info("👋 Cerrando MoodJournalAI API...")
```

**¿Qué hace?**
- Crea la app FastAPI
- Configura CORS (importante para frontend)
- Registra rutas (`/api/predict`)
- Añade endpoints de health check

---

### 3.5 - `backend/api/app/__init__.py`

```python
"""
MoodJournalAI API
API para clasificación de emociones con RoBERTa
"""
__version__ = "1.0.0"
```

---

### 3.6 - `backend/api/app/routes/__init__.py`

```python
"""Routes package"""
```

---

## 🚀 Paso 4: Ejecutar la API

### Comando para iniciar servidor

```bash
# Desde la raíz del proyecto
cd C:\MoodJournalAI

# Activar entorno virtual
.\.venv\Scripts\Activate

# Iniciar servidor FastAPI
uvicorn backend.api.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Parámetros:**
- `--reload`: Reinicia automáticamente al cambiar código (desarrollo)
- `--host 0.0.0.0`: Accesible desde cualquier IP
- `--port 8000`: Puerto 8000

**Output esperado:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     🚀 Iniciando MoodJournalAI API...
INFO:     Application startup complete.
```

---

## 🧪 Paso 5: Probar la API

### Opción 1: Swagger UI (Recomendado)

1. Abrir navegador: `http://localhost:8000/docs`
2. Ver documentación interactiva
3. Expandir `POST /api/predict`
4. Clic en "Try it out"
5. Escribir en el body:
   ```json
   {
     "text": "I feel so happy and excited today!"
   }
   ```
6. Clic "Execute"
7. Ver respuesta:
   ```json
   {
     "predicted_emotion": "joy",
     "confidence": 0.94,
     "all_scores": [
       {"emotion": "joy", "score": 0.94},
       {"emotion": "love", "score": 0.03},
       {"emotion": "surprise", "score": 0.02},
       {"emotion": "sadness", "score": 0.01},
       {"emotion": "anger", "score": 0.00},
       {"emotion": "fear", "score": 0.00}
     ]
   }
   ```

### Opción 2: curl

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am scared of the future\"}"
```

### Opción 3: Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={"text": "I feel so sad and lonely"}
)

result = response.json()
print(f"Emoción: {result['predicted_emotion']}")
print(f"Confianza: {result['confidence']:.2%}")
```

---

## 📊 Flujo Completo de una Request

```
1. Cliente HTTP
   ↓
   POST /api/predict
   Body: {"text": "I feel happy"}
   ↓
2. FastAPI recibe request
   ↓
3. Pydantic valida PredictionRequest
   ✅ text es string
   ✅ longitud entre 1-512
   ↓
4. predict_emotion() handler
   ↓
5. EmotionClassifier.predict(text)
   ↓
6. Tokenizar texto
   ↓
7. Modelo RoBERTa fine-tuned
   ↓
8. Softmax → Probabilidades
   ↓
9. Construir PredictionResponse
   ↓
10. FastAPI serializa a JSON
    ↓
11. Cliente recibe response
    {
      "predicted_emotion": "joy",
      "confidence": 0.94,
      "all_scores": [...]
    }
```

---

## ✅ Checklist de Implementación

- [ ] Actualizar `requirements.txt`
- [ ] Instalar FastAPI y uvicorn
- [ ] Crear estructura de carpetas `backend/api/`
- [ ] Implementar `models.py` (Pydantic)
- [ ] Implementar `ml_service.py` (carga modelo)
- [ ] Implementar `routes/predict.py` (endpoint)
- [ ] Implementar `main.py` (app FastAPI)
- [ ] Iniciar servidor con uvicorn
- [ ] Probar en Swagger UI `/docs`
- [ ] Probar con curl o Python
- [ ] Verificar que funciona correctamente

---

## 🎯 Próximos Pasos (Futuro)

### FASE 2: Streamlit App
- Crear interfaz visual simple
- Consumir la API FastAPI
- Mostrar predicciones con gráficos

### FASE 3: Visualización de Atención
- Modificar `ml_service.py` para extraer attention weights
- Nuevo endpoint `/api/predict/attention`
- Visualizar en frontend con D3.js

### FASE 4: Frontend React
- App moderna con React
- Visualización de atención interactiva
- Gráficos con Chart.js o Recharts

---

## 📝 Notas Importantes

1. **Primera request lenta**: El modelo se carga en la primera predicción (~2-5 segundos)
2. **Requests siguientes rápidas**: Modelo ya en memoria (~50-200ms)
3. **GPU detectada automáticamente**: Si tienes RTX 4060, la usará
4. **CORS habilitado**: Frontend puede consumir la API
5. **Documentación automática**: Siempre disponible en `/docs`

---

**Última actualización:** 2025-12-10
