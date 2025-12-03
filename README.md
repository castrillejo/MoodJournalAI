# MoodJournalAI 🧠💭

## 📋 Introducción

**MoodJournalAI** es un sistema inteligente de análisis de emociones y estados de ánimo basado en entradas de diario personal. El proyecto utiliza **procesamiento de lenguaje natural (NLP)** con modelos RoBERTa en inglés para analizar sentimientos en textos de diarios, identificando patrones emocionales y tendencias en el bienestar de los usuarios.

### 🎯 Características principales

- 🤖 **Modelo RoBERTa-base (inglés)** descargado localmente para análisis de sentimientos
- 🗄️ **Base de datos PostgreSQL** para almacenar entradas de diario
- 🔄 **Pipeline ETL** para carga de datos de muestra
- 📊 **Análisis de embeddings** con modelos transformer
- 🚀 Preparado para **fine-tuning** de modelos personalizados
- 🎭 **6 emociones detectables:** joy, sadness, fear, anger, love, surprise

---

## 📁 Estructura del Proyecto

```
MoodJournalAI/
├── backend/              # API backend (en desarrollo)
├── frontend/             # Interfaz de usuario (en desarrollo)
├── data/                 # Datos de muestra
│   ├── usuarios.csv      # Datos de usuarios (~7.8 KB)
│   └── entradas.csv      # Entradas de diario (~1.16 MB, 6,124 entradas)
├── etl/                  # Pipeline ETL
│   ├── load_data.py
│   ├── Dockerfile
│   └── requirements.txt
├── model-training/       # 🆕 Entrenamiento de modelos ML
│   ├── download-model/   # Scripts de descarga de modelos
│   │   ├── download_roberta.py
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── roberta-base-english/  # 🤖 Modelo RoBERTa (~500 MB)
│   │       ├── vocab.json
│   │       ├── merges.txt
│   │       ├── tokenizer.json
│   │       └── base/
│   │           └── model.safetensors
│   └── PLAN_FINETUNING.md
├── notebooks/            # 🆕 Jupyter notebooks y scripts de prueba
│   └── test_sentiment.py # Script de prueba de RoBERTa
├── docker/               # Configuraciones Docker
└── docker-compose.yml    # Orquestación de servicios
```

---

## 🚀 Instalación y Configuración

### 1️⃣ Requisitos Previos

- **Docker Desktop** (para base de datos)
- **Python 3.8+** (para modelos de ML)
- **Git** (para clonar el repositorio)

### 2️⃣ Entorno Virtual de Python

El proyecto utiliza un entorno virtual (`.venv`) para gestionar las dependencias de Python de forma aislada.

**Desde el directorio raíz del proyecto (`c:\MoodJournalAI>`):**

#### Activar el entorno virtual:
```powershell
.\.venv\Scripts\Activate
```

Una vez activado, verás `(.venv)` al inicio de tu prompt:
```
(.venv) c:\MoodJournalAI>
```

#### Desactivar el entorno virtual:
```powershell
deactivate
```

**💡 Nota:** Recuerda activar el entorno virtual antes de instalar dependencias o ejecutar scripts de Python relacionados con el proyecto.

### 3️⃣ Instalación de Docker

1. Descarga Docker Desktop desde [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Ejecuta el instalador y sigue las instrucciones
3. Reinicia tu computadora si es necesario
4. Verifica la instalación:
   ```powershell
   docker --version
   ```

### 4️⃣ Levantar la Base de Datos

Desde la raíz del proyecto:

```bash
# Construir y levantar los contenedores
docker-compose up --build

# O en segundo plano
docker-compose up --build -d
```

**Servicios disponibles:**
- PostgreSQL: `localhost:5432`
  - Usuario: `admin`
  - Contraseña: `admin`
  - Base de datos: `moodjournal`

**Comandos útiles de Docker:**
```bash
# Ver estado de los contenedores
docker-compose ps

# Ver logs
docker-compose logs -f

# Detener los contenedores
docker-compose down

# Conectarse a PostgreSQL
docker exec -it moodjournal_postgres psql -U admin -d moodjournal
```

---

## 🤖 Modelo RoBERTa-base para Análisis de Sentimientos

### 📥 Descarga del Modelo

El proyecto utiliza el modelo **RoBERTa-base** (inglés) optimizado para análisis de sentimientos.

**Características del modelo:**
- **Nombre:** `roberta-base`
- **Parámetros:** ~125 millones
- **Tamaño:** ~500 MB
- **Corpus:** BookCorpus, Wikipedia inglés, CC-News, OpenWebText, STORIES
- **Arquitectura:** 12 capas, 768 dimensiones, 12 attention heads
- **Idioma:** Inglés (optimizado para los textos del dataset)

**¿Por qué RoBERTa en lugar de BERT?**
- Mejor rendimiento en benchmarks de NLP
- Entrenamiento más robusto con más datos
- Optimizado para tareas de clasificación

#### Pasos para descargar:

```bash
# 1. Activar entorno virtual
.\.venv\Scripts\Activate

# 2. Ir a la carpeta de descarga
cd model-training/download-model

# 3. Instalar dependencias (si no están instaladas)
pip install -r requirements.txt

# 4. Ejecutar script de descarga
python download_roberta.py
```

El modelo se descargará en: `model-training/download-model/roberta-base-english/`

**⏱️ Tiempo estimado:** 3-10 minutos (dependiendo de tu conexión)

### 🧪 Probar el Modelo

Una vez descargado RoBERTa, puedes probarlo:

```bash
# Ir a la carpeta de notebooks
cd notebooks

# Ejecutar script de prueba
python test_sentiment.py
```

Este script:
- ✅ Carga el modelo RoBERTa desde tu carpeta local
- ✅ Tokeniza una frase de ejemplo en inglés
- ✅ Genera embeddings (representaciones numéricas de 768 dimensiones)
- ✅ Muestra las dimensiones del output

**Nota:** RoBERTa base solo genera embeddings. Para clasificar sentimientos en 6 emociones (joy, sadness, fear, anger, love, surprise), necesita fine-tuning.

---

## 🔧 Próximos Pasos

### En Desarrollo

- [ ] **Fine-tuning de RoBERTa** para clasificación de 6 emociones
- [ ] **Backend API** (FastAPI) para análisis de entradas
- [ ] **Frontend** (React/Next.js) para interfaz de usuario
- [ ] **Notebooks de análisis** exploratorio de datos
- [ ] **Sistema de evaluación** del modelo entrenado

### Roadmap

1. **Fase 1:** Preparación y preprocesamiento de datos de `entradas.csv` (6,124 entradas)
2. **Fase 2:** Fine-tuning de RoBERTa para 6 emociones personalizadas
3. **Fase 3:** Desarrollo de API backend
4. **Fase 4:** Desarrollo de interfaz frontend
5. **Fase 5:** Integración completa y deployment

---

## 📚 Recursos Adicionales

### Documentación por Módulo

- **model-training/download-model/README.md** - Guía completa de descarga de RoBERTa
- **model-training/PLAN_FINETUNING.md** - Plan detallado de fine-tuning
- **etl/README.md** - Pipeline ETL y carga de datos (próximamente)
- **backend/README.md** - API documentation (próximamente)
- **frontend/README.md** - UI documentation (próximamente)

### Tecnologías Utilizadas

- **NLP:** Hugging Face Transformers, PyTorch, RoBERTa-base
- **Base de datos:** PostgreSQL
- **Containerización:** Docker, Docker Compose
- **Backend (futuro):** FastAPI
- **Frontend (futuro):** React/Next.js

---

## 🤝 Contribuciones

Este proyecto está en desarrollo activo. Las contribuciones son bienvenidas.

---

## 📄 Licencia

[Especificar licencia]

---

## 👤 Autor

Asier Castrillejo - MoodJournalAI Project