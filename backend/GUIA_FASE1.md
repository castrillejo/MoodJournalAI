# Guía Rápida: Implementación Fase 1 - FastAPI

## 🎯 Objetivo
Tener una API REST funcionando que sirva predicciones de tu modelo RoBERTa

## ⏱️ Tiempo Estimado
**30-45 minutos** (si sigues los pasos exactos)

---

## 📋 Prerequisitos

✅ Modelo entrenado en `models/final/`  
✅ Entorno virtual activado (`.venv`)  
✅ Python 3.10+

---

## 🚀 Pasos de Implementación

### PASO 1: Instalar Dependencias (2 minutos)

```bash
# Activar entorno virtual
.\.venv\Scripts\Activate

# Instalar FastAPI y Uvicorn
pip install fastapi "uvicorn[standard]"
```

**Verificar instalación:**
```bash
python -c "import fastapi; import uvicorn; print('✅ Instalado correctamente')"
```

---

### PASO 2: Crear Estructura de Carpetas (1 minuto)

```bash
# Desde C:\MoodJournalAI

# Crear carpetas
mkdir backend\api\app\routes

# Crear archivos __init__.py
New-Item backend\api\app\__init__.py
New-Item backend\api\app\routes\__init__.py
```

**Resultado:**
```
backend/
└── api/
    └── app/
        ├── __init__.py
        └── routes/
            └── __init__.py
```

---

### PASO 3: Crear Archivos Python (20 minutos)

#### 3.1 - `backend/api/app/models.py`

```bash
New-Item backend\api\app\models.py
```

**📝 Copiar código del PLAN_FASTAPI.md sección 3.1**

#### 3.2 - `backend/api/app/ml_service.py`

```bash
New-Item backend\api\app\ml_service.py
```

**📝 Copiar código del PLAN_FASTAPI.md sección 3.2**

⚠️ **IMPORTANTE:** Verifica que la ruta al modelo sea correcta:
```python
model_path = Path(__file__).parent.parent.parent.parent / "models" / "final"
```

#### 3.3 - `backend/api/app/routes/predict.py`

```bash
New-Item backend\api\app\routes\predict.py
```

**📝 Copiar código del PLAN_FASTAPI.md sección 3.3**

#### 3.4 - `backend/api/app/main.py`

```bash
New-Item backend\api\app\main.py
```

**📝 Copiar código del PLAN_FASTAPI.md sección 3.4**

---

### PASO 4: Verificar Archivos Creados (1 minuto)

```bash
tree backend\api /F
```

**Deberías ver:**
```
backend\api
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── ml_service.py
│   ├── models.py
│   └── routes
│       ├── __init__.py
│       └── predict.py
```

---

### PASO 5: Iniciar Servidor (2 minutos)

```bash
# Desde C:\MoodJournalAI
uvicorn backend.api.app.main:app --reload --port 8000
```

**Output esperado:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     🚀 Iniciando MoodJournalAI API...
INFO:     Application startup complete.
```

**Si ves esto, ¡FUNCIONA! 🎉**

---

### PASO 6: Probar la API (10 minutos)

#### 6.1 - Abrir Swagger UI

1. Abrir navegador
2. Ir a: `http://localhost:8000/docs`
3. Deberías ver la documentación interactiva

**Captura de pantalla de lo que verás:**
```
MoodJournalAI API 1.0.0

Endpoints:
├── GET  /               (Root)
├── GET  /health         (Health check)
└── POST /api/predict    (Predict emotion) ← ESTE ES EL IMPORTANTE
```

#### 6.2 - Hacer Primera Predicción

1. En Swagger UI, expandir `POST /api/predict`
2. Clic en "Try it out"
3. Modificar el body:
   ```json
   {
     "text": "I feel so happy and excited about my new job!"
   }
   ```
4. Clic "Execute"
5. Scroll down para ver la respuesta

**Respuesta esperada (ejemplo):**
```json
{
  "predicted_emotion": "joy",
  "confidence": 0.94,
  "all_scores": [
    {"emotion": "joy", "score": 0.94},
    {"emotion": "love", "score": 0.03},
    {"emotion": "surprise", "score": 0.02},
    {"emotion": "sadness", "score": 0.01},
    {"emotion": "fear", "score": 0.00},
    {"emotion": "anger", "score": 0.00}
  ]
}
```

#### 6.3 - Probar Diferentes Textos

Prueba con:

1. **Texto de alegría:**
   ```
   "I'm thrilled about this opportunity!"
   ```
   Esperado: `joy`

2. **Texto de tristeza:**
   ```
   "I feel so lonely and sad"
   ```
   Esperado: `sadness`

3. **Texto de miedo:**
   ```
   "I'm scared of what might happen"
   ```
   Esperado: `fear`

4. **Texto de ira:**
   ```
   "I'm so angry and frustrated"
   ```
   Esperado: `anger`

5. **Texto de amor:**
   ```
   "I love spending time with you"
   ```
   Esperado: `love`

6. **Texto de sorpresa:**
   ```
   "Wow, I can't believe this happened!"
   ```
   Esperado: `surprise`

---

## ✅ Checklist de Verificación

- [ ] FastAPI instalado
- [ ] Estructura de carpetas creada
- [ ] Todos los archivos creados (4 archivos .py)
- [ ] Servidor inicia sin errores
- [ ] Swagger UI carga en `/docs`
- [ ] Primer prediction funciona
- [ ] Modelo carga correctamente
- [ ] Probado con diferentes emociones

---

## 🐛 Troubleshooting

### Error: "Modelo no encontrado"

**Causa:** Ruta incorrecta al modelo

**Solución:** Verificar en `ml_service.py` línea ~45:
```python
model_path = Path(__file__).parent.parent.parent.parent / "models" / "final"
print(f"Buscando modelo en: {model_path.absolute()}")
```

Ejecutar y verificar que la ruta sea correcta.

---

### Error: "ModuleNotFoundError: No module named 'fastapi'"

**Causa:** FastAPI no instalado

**Solución:**
```bash
pip install fastapi uvicorn[standard]
```

---

### Error: "Port 8000 already in use"

**Causa:** Otro servidor usando puerto 8000

**Solución:** Usar otro puerto:
```bash
uvicorn backend.api.app.main:app --reload --port 8001
```

---

### Error: CUDA out of memory (GPU)

**Causa:** Modelo muy grande para GPU

**Solución:** Usar CPU, modificar `ml_service.py`:
```python
# Forzar CPU
self._device = "cpu"
```

---

## 📊 Arquitectura de lo que Acabas de Crear

```
┌─────────────────┐
│   Cliente       │
│ (Navegador/curl)│
└────────┬────────┘
         │
         │ HTTP POST /api/predict
         │ {"text": "I feel happy"}
         │
         ▼
┌────────────────────┐
│    FastAPI         │
│   (main.py)        │
├────────────────────┤
│ 1. Valida request  │ ← models.py (Pydantic)
│ 2. Llama handler   │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  predict.py        │
│  (Endpoint)        │
├────────────────────┤
│ 3. Llama servicio  │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  ml_service.py     │
│  (Clasificador)    │
├────────────────────┤
│ 4. Tokeniza texto  │
│ 5. Modelo RoBERTa  │
│ 6. Softmax         │
└────────┬───────────┘
         │
         │ {"predicted_emotion": "joy", ...}
         │
         ▼
┌────────────────────┐
│    Response JSON   │
└────────────────────┘
```

---

## 🎯 ¿Qué Tienes Ahora?

✅ API REST funcionando en `http://localhost:8000`  
✅ Endpoint POST `/api/predict` que clasifica emociones  
✅ Documentación automática en `/docs`  
✅ Modelo RoBERTa sirviendo predicciones  
✅ Base para visualización de atención (futuro)  

---

## 🚀 Próximos Pasos

### Opción A: Streamlit App (Rápido)
Crear interfaz visual simple para probar el modelo

### Opción B: React Frontend (Profesional)
Crear frontend moderno que consume esta API

### Opción C: Visualización de Atención
Modificar API para extraer y visualizar attention weights

---

**¿Cuál prefieres hacer ahora?**
