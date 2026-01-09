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
│
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
- **Parámetros entrenables:** ~125 millones
- **Epochs:** 3
- **Accuracy:** ~77-79%
- **Uso:** Máxima precisión, requiere más recursos

### **2️⃣ Frozen (Feature Extraction)**
- **Estrategia:** RoBERTa congelado, solo Classification Head entrenable
- **Parámetros entrenables:** ~4,608 (0.003%)
- **Epochs:** 3
- **Accuracy:** ~65-70%
- **Uso:** Rápido, menos overfitting, embedded extraction

### **3️⃣ Semi-Frozen (Híbrido)**
- **Estrategia:** Congelar capas intermedias, entrenar primeras/últimas
- **Variantes disponibles:**
  - `semi_frozen2` → Descongela capas 0, 2
  - `semi_frozen4` → Descongela capas 0, 2, 4
  - `semi_frozen6` → Descongela capas 0, 2, 4, 6
- **Accuracy:** Variable según capas (~70-75%)
- **Uso:** Balance entre precisión y eficiencia

---

## 🌐 Endpoints de la API

### **Predicción**
- `POST /api/predict/single` → Predice emoción de un texto
- `POST /api/predict/batch` → Predice múltiples textos
- `POST /api/predict/attention` → Predice con attention weights

### **Evaluación**
- `GET /api/evaluation/overview` → Resumen de todos los modelos
- `GET /api/evaluation/test/{model}` → Métricas de un modelo específico
- `POST /api/evaluation/compare` → Compara predicciones

### **Usuarios**
- `GET /api/users` → Lista de usuarios
- `GET /api/users/{id}/entries` → Entradas de un usuario
- `GET /api/users/{id}/stats` → Estadísticas emocionales
- `GET /api/users/{id}/timeline` → Timeline de emociones

---

## 📊 Dashboard Frontend

### **Sección 1: Model Overview**
- Tabla comparativa de **Frozen / Semi-Frozen / Fine-tuned**
- Métricas: Accuracy, F1-Score, Loss
- Selector de variantes semi-frozen
- Indicadores visuales de rendimiento

### **Sección 2: Compare Predictions**
- Input de texto para clasificar
- Predicciones side-by-side de 3 modelos
- Confidence scores con barras de progreso
- **Visualización de Attention Weights** (heatmap)
- Permite cambiar entre variantes semi-frozen

### **Sección 3: Users Analysis**
- Lista de usuarios con preview de estadísticas
- Gráfica de distribución de emociones por usuario
- Timeline de emociones a lo largo del tiempo
- Tabla de entradas recientes con predicciones

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

## 📈 Próximos Pasos

### ✅ Completado
- [x] Fine-tuning de RoBERTa (3 estrategias)
- [x] Backend FastAPI completo
- [x] Frontend React con dashboard
- [x] Sistema de evaluación con métricas
- [x] Visualización de Attention Weights
- [x] Análisis por usuarios
- [x] Comparación de modelos

### 🚧 En Desarrollo
- [ ] Autenticación de usuarios
- [ ] Guardar predicciones en base de datos
- [ ] Export de reportes en PDF
- [ ] Modo oscuro completo
- [ ] Gráficos de evolución temporal

### 🎯 Futuro
- [ ] Deployment en cloud (AWS/GCP/Azure)
- [ ] Modelo multilenguaje (español)
- [ ] API pública con rate limiting
- [ ] App móvil (React Native)

---

## 📚 Documentación Adicional

- **[GUIA_PROYECTO.md](GUIA_PROYECTO.md)** - Guía paso a paso del proyecto
- **[SETUP_PC_CASA.md](SETUP_PC_CASA.md)** - Setup desde cero
- **[PLAN_FINETUNING.md](PLAN_FINETUNING.md)** - Teoría y plan de fine-tuning

---

## 🤝 Contribuciones

Este proyecto es parte de un trabajo académico. Las contribuciones son bienvenidas.

---

## 📄 Licencia

[Especificar licencia]

---

## 👤 Autor

**Asier Castrillejo**  
MoodJournalAI Project - 2025  
*Sistema de clasificación de emociones usando RoBERTa fine-tuning*