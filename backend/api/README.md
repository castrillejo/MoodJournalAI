# MoodJournalAI - FastAPI Backend

API REST para clasificación de emociones usando RoBERTa fine-tuned.

## 🚀 Inicio Rápido

### 1. Activar entorno virtual
```powershell
cd C:\MoodJournalAI
.\.venv\Scripts\Activate
```

### 2. Iniciar servidor
```powershell
uvicorn backend.api.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Acceder a la documentación
Abre tu navegador en: **http://localhost:8000/docs**

## 📡 Endpoints

### `GET /`
Información básica de la API
```bash
curl http://localhost:8000/
```

### `GET /health`
Health check
```bash
curl http://localhost:8000/health
```

### `POST /api/predict`
Predice la emoción de un texto

**Request:**
```json
{
  "text": "I feel so happy today!"
}
```

**Response:**
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

## 🧪 Probar la API

### Opción 1: Swagger UI (Recomendado)
1. Ve a http://localhost:8000/docs
2. Expande `POST /api/predict`
3. Click "Try it out"
4. Ingresa un texto
5. Click "Execute"

### Opción 2: curl
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I am so excited about this project!\"}"
```

### Opción 3: Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/predict",
    json={"text": "I feel sad and lonely"}
)

result = response.json()
print(f"Emoción: {result['predicted_emotion']}")
print(f"Confianza: {result['confidence']:.2%}")
```

## 📊 Emociones Soportadas

- `joy` (alegría)
- `sadness` (tristeza)
- `fear` (miedo)
- `anger` (ira)
- `love` (amor)
- `surprise` (sorpresa)

## ⚙️ Configuración

### Puerto personalizado
```powershell
uvicorn backend.api.app.main:app --port 5000
```

### Sin auto-reload (producción)
```powershell
uvicorn backend.api.app.main:app --host 0.0.0.0 --port 8000
```

## 📝 Notas

- **Primera request lenta**: El modelo se carga en la primera predicción (~2-5 segundos si hay GPU)
- **Requests siguientes**: ~50-200ms con GPU
- **GPU auto-detectada**: Si tienes RTX 4060, la usará automáticamente
- **CORS habilitado**: El frontend puede consumir la API desde cualquier origen

## 🔧 Troubleshooting

### Error: Modelo no encontrado
Verifica que el modelo fine-tuned esté en:
```
C:\MoodJournalAI\model-training\download-model\roberta-base-english\finetuned-emotion\
```

### Puerto ocupado
Si el puerto 8000 está ocupado, usa otro:
```powershell
uvicorn backend.api.app.main:app --port 8001
```

## 📚 Documentación Adicional

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
