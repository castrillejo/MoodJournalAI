# MoodJournalAI 🧠💭

## 📋 Introducción

**MoodJournalAI** es un sistema inteligente de análisis de emociones y estados de ánimo basado en entradas de diario personal. El proyecto utiliza **procesamiento de lenguaje natural (NLP)** con modelos BERT en español para analizar sentimientos en textos de diarios, identificando patrones emocionales y tendencias en el bienestar de los usuarios.

### 🎯 Características principales

- 🤖 **Modelo BERT en español (BETO)** descargado localmente para análisis de sentimientos
- 🗄️ **Base de datos PostgreSQL** para almacenar entradas de diario
- 🔄 **Pipeline ETL** para carga de datos de muestra
- 📊 **Análisis de embeddings** con modelos transformer
- 🚀 Preparado para **fine-tuning** de modelos personalizados

---

## 📁 Estructura del Proyecto

```
MoodJournalAI/
├── backend/              # API backend (en desarrollo)
├── frontend/             # Interfaz de usuario (en desarrollo)
├── data/                 # Datos de muestra
│   ├── usuarios.csv      # Datos de usuarios (~7.8 KB)
│   └── entradas.csv      # Entradas de diario (~1.16 MB)
├── etl/                  # Pipeline ETL
│   ├── load_data.py
│   ├── Dockerfile
│   └── requirements.txt
├── model-training/       # 🆕 Entrenamiento de modelos ML
│   └── download-model/   # Scripts de descarga de modelos
│       ├── download_beto.py
│       ├── requirements.txt
│       ├── README.md
│       └── bert-base-spanish/  # 🤖 Modelo BETO (~440 MB)
│           ├── vocab.txt
│           ├── tokenizer.json
│           └── base/
│               └── model.safetensors
├── notebooks/            # 🆕 Jupyter notebooks y scripts de prueba
│   └── test_sentiment.py # Script de prueba de BETO
├── docker/               # Configuraciones Docker
└── docker-compose.yml    # Orquestación de servicios
```

---

## 🚀 Instalación y Configuración

### 1️⃣ Requisitos Previos

- **Docker Desktop** (para base de datos)
- **Python 3.8+** (para modelos de ML)
- **Git** (para clonar el repositorio)

### 2️⃣ Instalación de Docker

1. Descarga Docker Desktop desde [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Ejecuta el instalador y sigue las instrucciones
3. Reinicia tu computadora si es necesario
4. Verifica la instalación:
   ```powershell
   docker --version
   ```

### 3️⃣ Levantar la Base de Datos

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

## 🤖 Modelo BERT (BETO) para Análisis de Sentimientos

### 📥 Descarga del Modelo

El proyecto incluye el modelo **BETO** (BERT base entrenado en español) de la Universidad de Chile.

**Características del modelo:**
- **Nombre:** `dccuchile/bert-base-spanish-wwm-cased`
- **Parámetros:** ~110 millones
- **Tamaño:** ~440 MB
- **Corpus:** Wikipedia español, libros, noticias
- **Arquitectura:** 12 capas, 768 dimensiones, 12 attention heads

#### Pasos para descargar:

```bash
# 1. Ir a la carpeta de descarga
cd model-training/download-model

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar script de descarga
python download_beto.py
```

El modelo se descargará en: `model-training/download-model/bert-base-spanish/`

**⏱️ Tiempo estimado:** 3-10 minutos (dependiendo de tu conexión)

### 🧪 Probar el Modelo

Una vez descargado BETO, puedes probarlo:

```bash
# Ir a la carpeta de notebooks
cd notebooks

# Ejecutar script de prueba
python test_sentiment.py
```

Este script:
- ✅ Carga el modelo BETO desde tu carpeta local
- ✅ Tokeniza una frase de ejemplo
- ✅ Genera embeddings (representaciones numéricas)
- ✅ Muestra las dimensiones del output

**Nota:** BETO base solo genera embeddings. Para clasificar sentimientos (POS/NEG/NEU), necesita fine-tuning.

---

## 🔧 Próximos Pasos

### En Desarrollo

- [ ] **Fine-tuning de BETO** para clasificación de sentimientos multi-emoción
- [ ] **Backend API** (FastAPI) para análisis de entradas
- [ ] **Frontend** (React/Next.js) para interfaz de usuario
- [ ] **Notebooks de análisis** exploratorio de datos
- [ ] **Sistema de etiquetado** de datos para entrenamiento

### Roadmap

1. **Fase 1:** Preparación y etiquetado de datos de `entradas.csv`
2. **Fase 2:** Fine-tuning de BETO para sentimientos personalizados
3. **Fase 3:** Desarrollo de API backend
4. **Fase 4:** Desarrollo de interfaz frontend
5. **Fase 5:** Integración completa y deployment

---

## 📚 Recursos Adicionales

### Documentación por Módulo

- **model-training/download-model/README.md** - Guía completa de descarga de modelos
- **etl/README.md** - Pipeline ETL y carga de datos (próximamente)
- **backend/README.md** - API documentation (próximamente)
- **frontend/README.md** - UI documentation (próximamente)

### Tecnologías Utilizadas

- **NLP:** Hugging Face Transformers, PyTorch
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