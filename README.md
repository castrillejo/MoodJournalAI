# MoodJournalAI 🧠💭

## 📋 Introducción

**MoodJournalAI** es un sistema inteligente de análisis de emociones y estados de ánimo basado en entradas de diario personal. El proyecto utiliza **procesamiento de lenguaje natural (NLP)** con modelos RoBERTa fine-tuned para clasificar emociones en textos de diarios, proporcionando análisis detallados con visualizaciones de attention mechanisms y comparación entre diferentes estrategias de entrenamiento.

### 🎯 Características principales

#### 🤖 **Modelos de ML Entrenados**
- **RoBERTa Fine-tuned Completo** 
- **RoBERTa Frozen (Feature Extraction)** 
- **RoBERTa Semi-Frozen (Capas 0, 2, 4, 6)** 
- **Visualización de Attention Weights** 

#### 💻 **Aplicación Full-Stack Funcional**
- **Frontend React** con Vite + TailwindCSS + Framer Motion
- **Backend FastAPI** con modelos ML integrados
- **Dashboard Interactivo** con métricas, gráficos y comparaciones
- **Sección de Usuarios** para análisis de patrones emocionales por usuario

#### 📊 **Análisis y Evaluación**
- 🎭 **6 emociones clasificables:** joy, sadness, fear, anger, love, surprise
- 📈 **Métricas completas:** Accuracy, Precision, Recall, F1-Score
- 🧩 **Matriz de Confusión** para análisis detallado
- 🔍 **Comparación de modelos** side-by-side

#### 🗄️ **Infraestructura**
- PostgreSQL con Docker Compose
- Pipeline ETL para carga de datos
- 6,124+ entradas de diario etiquetadas

---

## 🚀 Guía de Arranque Rápida

> **💻 ¿Necesitas transferir el proyecto a otro PC?**  
> 👉 **[Ver GUIA_MIGRACION_PORTATIL.md](GUIA_MIGRACION_PORTATIL.md)** - Guía completa para migrar vía pendrive

Para ejecutar la aplicación completa, necesitas **3 servicios en paralelo**:

### **Terminal 1: Base de Datos** 🗄️
```powershell
# Desde la raíz del proyecto
docker-compose up -d
```
*PostgreSQL corriendo en `localhost:5432`*

---

### **Terminal 2: Backend API** 🧠
```powershell
# 1. Activar entorno virtual
.\.venv\Scripts\Activate

# 2. Iniciar servidor FastAPI
python -m uvicorn backend.api.app.main:app --reload
```
*API corriendo en `http://127.0.0.1:8000`*  
*Documentación interactiva: `http://127.0.0.1:8000/docs`*

---

### **Terminal 3: Frontend Web** 💻
```powershell
# 1. Entrar en la carpeta frontend
cd frontend

# 2. Instalar dependencias (solo primera vez)
npm install

# 3. Iniciar aplicación React
npm run dev
```
*Aplicación web en `http://localhost:5173`*

---

### **✅ Verificación**

Abre tu navegador en **`http://localhost:5173`** y deberías ver:
- ✅ **API: Online** (indicador verde en la cabecera)
- ✅ **Model Overview** con métricas de los 3 modelos
- ✅ **Compare Predictions** para probar predicciones
- ✅ **Users Analysis** con patrones emocionales por usuario

---

## 📁 Estructura del Proyecto

```
MoodJournalAI/
├── 📄 README.md                    # Este archivo
├── 📂 backend/                     # ✅ BACKEND FASTAPI (COMPLETO)
│   ├── api/
│   │   └── app/
│   │       ├── main.py             # Aplicación FastAPI principal
│   │       ├── ml_service.py       # Carga de modelos ML
│   │       ├── analysis_service.py # Análisis de usuarios
│   │       ├── models.py           # Schemas Pydantic
│   │       └── routes/
│   │           ├── predict.py      # Endpoints de predicción
│   │           ├── evaluation.py   # Endpoints de métricas
│   │           └── users.py        # Endpoints de usuarios
│   └── requirements.txt
│
├── 📂 frontend/                    # ✅ FRONTEND REACT (COMPLETO)
│   ├── src/
│   │   ├── App.jsx                 # Componente principal
│   │   ├── components/
│   │   │   ├── ModelOverviewSection.jsx
│   │   │   ├── ComparePredictionsSection.jsx
│   │   │   ├── UsersSection.jsx
│   │   │   ├── AttentionVisualization.jsx
│   │   │   ├── EmotionChart.jsx
│   │   │   ├── MetricsPanel.jsx
│   │   │   ├── ConfusionMatrixGrid.jsx
│   │   │   └── ... (10 componentes)
│   │   └── services/
│   │       └── api.js              # Cliente API
│   ├── package.json
│   └── tailwind.config.js
│
├── 📂 model-training/              # ✅ ML TRAINING (COMPLETO)
│   ├── data/
│   │   ├── train.csv               # 4,900 entradas
│   │   ├── val.csv                 # 610 entradas
│   │   └── test.csv                # 614 entradas
│   │
│   ├── models/                     # Modelos entrenados
│   │   ├── checkpoints/            # Fine-tuned completo
│   │   ├── checkpoints-frozen/     # Feature extraction
│   │   ├── checkpoints-semi-frozen2/
│   │   ├── checkpoints-semi-frozen4/
│   │   └── checkpoints-semi-frozen6/
│   │
│   ├── logs/                       # TensorBoard logs
│   │   ├── finetuned/
│   │   ├── frozen/
│   │   └── semi_frozen*/
│   │
│   └── download-model/
│       └── roberta-base-english/   # Modelo RoBERTa base (~500 MB)
│
├── 📂 notebooks/                   # Scripts de entrenamiento
│   ├── train.py                    # Fine-tuning completo
│   ├── train_frozen.py             # Feature extraction
│   ├── train_semi_frozen.py        # Semi-frozen (configurable)
│   └── evaluation.py               # Evaluación y métricas
│
├── 📂 data/                        # Datos originales
│   ├── entradas.csv                # 6,124 entradas etiquetadas
│   ├── usuarios.csv                # Información de usuarios
│   └── finetuning/                 # Train/val/test splits
│
├── 📂 etl/                         # Pipeline ETL
│   ├── load_data.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── 📂 docker/                      # Configuración Docker
│   └── db/init/
│
├── docker-compose.yml              # Orquestación PostgreSQL + ETL
└── .venv/                          # Entorno virtual Python
```

---

## 🎓 Modelos Entrenados

### **1️⃣ Fine-tuned Completo** (Baseline)
- **Estrategia:** Todas las capas de RoBERTa + Classification Head entrenables
- **Epochs:** 3
- **Accuracy:** ~90.9%

### **2️⃣ Frozen (Feature Extraction)**
- **Estrategia:** RoBERTa congelado, solo Classification Head entrenable
- **Epochs:** 3
- **Accuracy:** 48.9%

### **3️⃣ Semi-Frozen (Híbrido)**
- **Estrategia:** Congelar capas intermedias, entrenar primeras/últimas
- **Variantes disponibles:**
  - `semi_frozen2` → Descongela capas 0, 2
  - `semi_frozen4` → Descongela capas 0, 2, 4
  - `semi_frozen6` → Descongela capas 0, 2, 4, 6
- **Accuracy:** Variable según capas (~75-85%)

---

## 🌐 Endpoints de la API

### **Predicción**
- `POST /api/predict/attention` → Predice emoción con attention weights (permite elegir modelo: finetuned/frozen/semi_frozenX)
- `POST /api/predict/compare/attention` → Compara predicciones de 3 modelos (frozen, semi, finetuned) con attention

### **Evaluación**
- `GET /api/evaluation/overview` → Resumen completo de todos los modelos con métricas y variantes semi-frozen

### **Usuarios**
- `GET /api/users/search?q=query&limit=5` → Búsqueda de usuarios por nombre 
- `GET /api/users/{user_id}/stats` → Estadísticas completas del usuario (análisis emocional + gráficos)

---

## 📊 Dashboard Frontend

### **Sección 1: Model Overview**
- Tabla comparativa de **Frozen / Semi-Frozen / Fine-tuned**
- Métricas: Accuracy, F1-Score, Loss
- Permite cambiar entre variantes semi-frozen

### **Sección 2: Compare Predictions**
- Input de texto para clasificar
- Predicciones side-by-side de 3 modelos
- Confidence scores con barras de progreso
- **Visualización de Attention Weights**

### **Sección 3: Users Analysis**
- Barra de busqueda de usuarios
- Gráfica de distribución de emociones por usuario
- Emoción principal
- Diferentes estadísticas complementarias sobre el usuario

---

## 🛠️ Tecnologías Utilizadas

### **Machine Learning**
- **Hugging Face Transformers** → RoBERTa, tokenizers
- **PyTorch** → Training framework
- **Scikit-learn** → Métricas y evaluación
- **Datasets** → Data loading

### **Backend**
- **FastAPI** → API REST moderna
- **Uvicorn** → ASGI server
- **Pydantic** → Validación de datos
- **Python 3.10+**

### **Frontend**
- **React 19** → UI framework
- **Vite** → Build tool
- **TailwindCSS 4** → Styling
- **Framer Motion** → Animaciones
- **Recharts** → Gráficos
- **Lucide React** → Íconos
- **Axios** → HTTP client

### **Infraestructura**
- **Docker + Docker Compose** → Contenedores
- **PostgreSQL 15** → Base de datos
- **NVIDIA CUDA** → GPU acceleration (opcional)

---

## 🔧 Comandos Útiles

### **Backend**
```powershell
# Activar entorno virtual
.\.venv\Scripts\Activate

# Instalar dependencias
pip install -r backend/requirements.txt

# Ejecutar con hot-reload
python -m uvicorn backend.api.app.main:app --reload

# Ver documentación interactiva
# http://127.0.0.1:8000/docs
```

### **Frontend**
```powershell
# Instalar dependencias
cd frontend
npm install

# Desarrollo
npm run dev

# Build para producción
npm run build

# Preview de producción
npm run preview
```

### **Docker**
```powershell
# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down

# Conectar a PostgreSQL
docker exec -it moodjournal_postgres psql -U admin -d moodjournal
```

### **Training**
```powershell
# Fine-tuning completo
python notebooks/train.py

# Frozen
python notebooks/train_frozen.py

# Semi-frozen (editar capas en el script)
python notebooks/train_semi_frozen.py

# Evaluación
python notebooks/evaluation.py
```

---

## 🎓 Cómo Entrenar un Modelo Nuevo

Si quieres entrenar un nuevo modelo desde cero usando el modelo base de RoBERTa, sigue estos pasos:

### **📋 Prerequisitos**

Asegúrate de tener:

1. ✅ **Modelo base RoBERTa descargado**
   - Ubicación: `model-training/download-model/roberta-base-english/base/`
   - Archivos: `model.safetensors`, `config.json`, tokenizer files

2. ✅ **Datos preparados** en splits train/val/test
   - Ubicación: `data/finetuning/`
   - Archivos: `train.csv`, `val.csv`, `test.csv`
   - Columnas requeridas: `texto_diario`, `emocion_principal`

3. ✅ **Entorno virtual activado** con dependencias instaladas
   ```powershell
   .\.venv\Scripts\Activate
   pip install -r backend/requirements.txt
   ```

---

### **🚀 Paso 1: Entrenar el Modelo**

Tienes 3 opciones de entrenamiento:

#### **Opción A: Fine-tuning Completo** (Recomendado)
Entrena todas las capas del modelo para máxima precisión.

```powershell
# Desde la raíz del proyecto
python notebooks/train.py
```

**Interacción:**
```
Iniciando configuración de Fine-Tuning...
📂 Cargando modelo base desde: model-training\download-model\roberta-base-english\base
Modelo y Tokenizer cargados.
Datos cargados: 4899 train, 612 validation
Tokenizando datos...

¿Quieres comenzar el entrenamiento AHORA? (s/n): s
```

**Responde `s`** para iniciar el entrenamiento.

---

#### **Opción B: Frozen (Feature Extraction)**
Solo entrena el classification head, RoBERTa permanece congelado.

```powershell
python notebooks/train_frozen.py
```

Más rápido pero menor precisión (~48-50% accuracy).

---

#### **Opción C: Semi-Frozen**
Entrena selectivamente ciertas capas de RoBERTa.

```powershell
python notebooks/train_semi_frozen.py
```

Necesitas editar el script para elegir qué capas descongelar (0, 2, 4, 6, etc.).

---

### **⏱️ Duración del Entrenamiento**

- **Con GPU (NVIDIA CUDA):** 15-30 minutos (3 epochs)
- **Con CPU:** 2-4 horas (puede ser más lento)

El script detecta automáticamente si tienes GPU disponible y usa fp16 para acelerar.

---

**Archivos generados:**

```
model-training/
├── models/
│   └── checkpoints/              # Checkpoints por epoch
│       ├── checkpoint-306/
│       ├── checkpoint-612/
│       └── checkpoint-918/
│
├── logs/                         # TensorBoard logs
│   └── events.out.tfevents...
│
└── download-model/
    └── roberta-base-english/
        └── finetuned-emotion/    # 🎯 MODELO FINAL
            ├── config.json
            ├── model.safetensors
            ├── training_args.bin
            └── ... (archivos del modelo)
```

---

### **📊 Paso 2: Evaluar el Modelo**

Una vez entrenado, evalúa el modelo en el test set:

```powershell
python notebooks/evaluation.py
```

**Output esperado:**

```
Se está haciendo la evaluación del modelo: finetuned...
OK. Guardado: backend\api\app\assets\evaluation\report_finetuned.json

Se está haciendo la evaluación del modelo: frozen...
OK. Guardado: backend\api\app\assets\evaluation\report_frozen.json

Se está haciendo la evaluación de los modelos: semi_frozen2/4/6...
OK. Guardado: backend\api\app\assets\evaluation\report_semi_frozen.json
```

**Archivos generados:**

```
backend/api/app/assets/evaluation/
├── report_finetuned.json    # Métricas del modelo fine-tuned
├── report_frozen.json        # Métricas del modelo frozen
└── report_semi_frozen.json   # Métricas de variantes semi-frozen
```

Estos JSON son **automáticamente cargados por la API** para mostrar las métricas en el frontend.

---


### **✅ Checklist de Entrenamiento**

- [ ] Modelo base RoBERTa descargado en `/base/`
- [ ] Datos en `data/finetuning/` (train.csv, val.csv, test.csv)
- [ ] Entorno virtual activado
- [ ] Dependencias instaladas
- [ ] Ejecutado `python notebooks/train.py` → Respondido 's'
- [ ] Entrenamiento completado sin errores
- [ ] Modelo guardado en `finetuned-emotion/`
- [ ] Ejecutado `python notebooks/evaluation.py`
- [ ] Generados JSONs en `backend/api/app/assets/evaluation/`
- [ ] Backend reiniciado
- [ ] Métricas visibles en frontend

---

**¡Listo! Ya tienes un modelo nuevo entrenado y funcionando. 🎉**
**Si ya entrenaste modelos con anterioridad, tener cuidado con archivos duplicados o con generar nuevos con nuevos nombres y rutas**

## 👤 Autor

**Asier Castrillejo**  
MoodJournalAI Project - 2025  
*Sistema de clasificación de emociones usando RoBERTa fine-tuning*