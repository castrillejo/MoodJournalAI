# Mapa del Proyecto MoodJournalAI 🗺️

Este documento ofrece una visión detallada de la estructura actual del proyecto, las tecnologías utilizadas y el estado de los diferentes componentes. Ha sido generado para facilitar la reincorporación al desarrollo tras un periodo de inactividad.

## 🏗️ Estructura del Proyecto

```text
MoodJournalAI/
├── 📂 .venv/                 # Tu entorno virtual de Python (contiene las librerías instaladas).
├── 📂 backend/               # Código del servidor.
│   ├── 📂 api/               # API principal construida con FastAPI.
│   │   ├── 📂 app/           # Lógica de la aplicación.
│   │   │   ├── 📂 routes/    # Endpoints (ej. predicciones).
│   │   │   └── ml_service.py # Servicio que carga el modelo de IA.
│   ├── 📂 web/               # Prototipo rápido de UI.
│   │   └── streamlit_app.py  # Aplicación Streamlit para probar el modelo sin el frontend completo.
├── 📂 data/                  # Datos del proyecto.
│   ├── 📂 finetuning/        # Datasets YA procesados (train.csv, val.csv, test.csv).
│   ├── entradas.csv          # Dataset original (fuente).
│   └── usuarios.csv          # Datos de usuarios.
├── 📂 docker/                # Configuración de contenedores (Base de datos).
├── 📂 etl/                   # Scripts para Extract-Transform-Load.
│   └── load_data.py          # Script para poblar la BBDD desde los CSV.
├── 📂 frontend/              # Interfaz de usuario final (React + Vite).
├── 📂 model-training/        # Núcleo de Inteligencia Artificial.
│   ├── 📂 download-model/    # Scripts para descargar el modelo base RoBERTa.
│   ├── 📂 logs/              # Registros de TensorBoard de entrenamientos previos.
│   ├── 📂 models/            # Checkpoints guardados.
│   │   ├── 📂 checkpoints/        # Modelos de entrenamiento completo.
│   │   └── 📂 checkpoints-frozen/ # Modelos de entrenamiento con capas congeladas.
│   ├── 📂 results/           # Gráficos y métricas de evaluación.
│   ├── train.py              # Script principal de entrenamiento.
│   └── evaluate.py           # Script de evaluación.
├── 📂 notebooks/             # Scripts de prueba rápida (ej. test_sentiment.py).
├── docker-compose.yml        # Orquestador para levantar la base de datos PostgreSQL.
├── SETUP_PC_CASA.md          # Guía de instalación inicial.
└── README.md                 # Archivo de introducción original.
```

---

## 🛠️ Tecnologías y Herramientas

### Backend & AI
- **FastAPI** (`backend/api`): Framework de API de alto rendimiento. Se encarga de recibir textos del frontend y devolver las emociones detectadas.
- **Streamlit** (`backend/web`): Herramienta para crear "dashboards" de datos rápidamente. Se usa aquí para testear el modelo visualmente sin depender del desarrollo del frontend de React.
- **RoBERTa (Hugging Face)**: El "cerebro" del proyecto. Un modelo Transformer pre-entrenado que estamos adaptando (fine-tuning) para detectar 6 emociones específicas.
- **PyTorch**: La librería de Deep Learning que mueve todo el entrenamiento e inferencia.

### Frontend
- **React + Vite** (`frontend`): La tecnología elegida para la web final. Rápida y moderna.
- **Tailwind CSS**: Framework de estilos para diseñar rápido sin salir del HTML.

### Infraestructura
- **Docker & Docker Compose**: Se usan principalmente para "encapsular" la base de datos PostgreSQL. Esto evita que tengas que instalar y configurar Postgres manualmente en tu Windows.
- **PostgreSQL**: Donde se guardan las entradas del diario a largo plazo.

---

## 🔍 Estado de las Carpetas "Clave"
Para que sepas dónde te quedaste:

1.  **`model-training/models/`**:
    *   **Estado**: No está vacía. Contiene carpetas `checkpoints` y `checkpoints-frozen`. Esto indica que **ya has ejecutado entrenamientos anteriormente**. Deberías tener modelos parciales o finales guardados ahí.

2.  **`data/finetuning/`**:
    *   **Estado**: Contiene `train.csv`, `val.csv` y `test.csv`.
    *   **Significado**: El script `prepare_dataset.py` ya se ejecutó con éxito. Los datos están listos para ser usados por los scripts de entrenamiento sin necesidad de preprocesarlos de nuevo.

3.  **`backend/api/`**:
    *   Contiene una estructura con `routes`, `models.py` y `ml_service.py`. Parece que el esqueleto de la API está listo para conectar con el modelo.

---

## 🚀 Cómo Retomar el Trabajo

1.  **Activa el entorno**:
    ```powershell
    .\.venv\Scripts\Activate
    ```

2.  **Levanta la Base de Datos**:
    ```powershell
    docker-compose up -d
    ```

3.  **Revisa tus modelos**:
    Como tienes checkpoints en `model-training/models`, podrías intentar evaluarlos:
    ```powershell
    python model-training/evaluate.py
    ```

4.  **Si quieres probar la web rápida (Streamlit)**:
    ```powershell
    streamlit run backend/web/streamlit_app.py
    ```

Este archivo (`PROJECT_MAP.md`) puede servirte de referencia rápida mientras navegas por el código.
