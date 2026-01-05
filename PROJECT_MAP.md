# Mapa del Proyecto MoodJournalAI (Estado Actual) 🗺️

Este documento refleja el estado real del proyecto, donde ya tienes una **aplicación funcional "Full Stack"** capaz de realizar predicciones y visualizar la atención del modelo.

## 🟢 Estado Actual: Funcional

A diferencia de un proyecto vacío, actualmente tienes:
1.  **Frontend (React)**: Una interfaz moderna y animada que permite escribir texto, enviarlo al servidor y ver la emoción predicha y los mapas de atención.
2.  **Backend (FastAPI)**: Un servidor levantado que recibe estas peticiones y utiliza el modelo de IA para responder.
3.  **Conexión**: Ambos extremos están conectados. El frontend chequea automáticamente si el backend está online (`/health`).

---

## 🏗️ Estructura Detallada

### 🖥️ Frontend (`/frontend`)
*Estado: Avanzado / Funcional*

Es una SPA (Single Page Application) construida con React y Vite.
- **`App.jsx`**: Controlador principal. Gestiona el estado de la API (`online`/`offline`) y orquesta la vista.
- **`services/api.js`**: Cliente HTTP (Axios) configurado para hablar con `localhost:8000`.
    - `predictEmotion(text)`: Envía texto simple.
    - `predictEmotionWithAttention(text)`: Solicita también los pesos de atención (para ver en qué palabras se fija el modelo).
- **`components/`**:
    - `TextInput.jsx`: Caja de texto con ejemplos rápidos y toggle para "Attention Weights".
    - `ResultCard.jsx`: Muestra la emoción ganadora y probabilidades.
    - `AttentionVisualization.jsx`: Renderiza gráficamente qué palabras pesaron más en la decisión.

### 🔌 Backend (`/backend`)
*Estado: Funcional / Sirviendo API*

- **`api/app/main.py`**: Punto de entrada. Monta las rutas y CORS.
- **`api/app/routes/predict.py`**: Define los endpoints `/predict` y `/predict/attention`.
- **`api/app/ml_service.py`**: Carga el modelo (RoBERTa) en memoria y ejecuta la inferencia real.

### 🧠 Inteligencia Artificial (`/model-training`)
*Estado: En proceso de mejora (Fine-tuning)*

- Tienes checkpoints de entrenamientos previos en `models/checkpoints`.
- El sistema actual usa estos modelos (o el base) para las predicciones que ves en el frontend.

---

## 🚀 Cómo volver a "Tu Sesión Anterior"

Para recuperar el entorno donde hacías pruebas en el navegador:

### 1. Levanta el Backend (Cerebro)
Necesitas una terminal para esto.
```powershell
# Activa el entorno (si no lo está)
.\.venv\Scripts\Activate

# Ve a la carpeta de la API
cd backend/api

# Lanza el servidor (deja esta terminal abierta)
uvicorn app.main:app --reload
```
*Debería decirte que está corriendo en `http://127.0.0.1:8000`*.

### 2. Levanta el Frontend (Cara)
Abre **otra** terminal nueva.
```powershell
cd frontend

# Instala dependencias (solo si hace mucho que no tocas nada, por seguridad)
npm install

# Lanza la web
npm run dev
```
*Abrirá tu navegador en `http://localhost:5173` (o te dará el link)*.

### 3. ¡Prueba!
Ve a `http://localhost:5173`. Deberías ver:
- El indicador de **API: Online** (verde) arriba a la derecha.
- La caja para escribir "I feel..."
- El botón "Analyze Emotion".

---

## 📂 Resumen de Archivos Clave

| Archivo | Propósito |
| -- | -- |
| `frontend/src/App.jsx` | Lógica visual principal. Si quieres cambiar colores o textos de la web, es aquí. |
| `backend/api/app/routes/predict.py` | Si quieres cambiar qué devuelve la API (ej. más datos). |
| `model-training/train.py` | Si decides volver a entrenar el modelo para que detecte mejor las emociones. |
| `docker-compose.yml` | Base de datos (necesaria si guardas historial, aunque la predicción pura a veces funciona sin ella si solo usa RAM). |

---

Este mapa sustituye al anterior para reflejar que ya tienes un producto mínimo viable (MVP) funcionando.
