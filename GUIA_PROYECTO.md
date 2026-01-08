# 📘 Guía Completa del Proyecto MoodJournalAI

Este documento detalla la estructura actual del proyecto, las tecnologías empleadas y, lo más importante, cómo ponerlo en marcha paso a paso.

## 🗺️ Mapa del Proyecto

El sistema está dividido en tres partes principales que deben funcionar simultáneamente:

```text
MoodJournalAI/
├── 📂 backend/               # 🧠 CEREBRO (API Python/FastAPI)
│   ├── api/routes/           # Endpoints (dónde llegan las peticiones)
│   └── ml_service.py         # Carga el modelo de IA
├── 📂 frontend/              # 💻 CARA (Web React/Vite)
│   ├── src/components/       # Botones, textos, gráficos
│   └── src/services/         # Conecta con el Cerebro
├── 📂 data/                  # 💾 MEMORIA (Datos CSV)
├── 📂 docker/                # 🐳 INFRAESTRUCTURA (Base de datos)
├── 📂 model-training/        # 🏋️ ENTRENAMIENTO (Scripts de ML)
└── .venv/                    # Librerías de Python instaladas
```

---

## 🛠️ Tecnologías

1.  **Backend (FastAPI)**: Gestiona la lógica y la IA. Escucha en el puerto `8000`.
2.  **Frontend (React + Vite)**: Lo que ves en el navegador. Corre en el puerto `5173`.
3.  **Base de Datos (PostgreSQL)**: Se ejecuta dentro de Docker.
4.  **IA (RoBERTa)**: El modelo que detecta emociones.

---

## ▶️ Cómo Ejecutar el Proyecto (Paso a Paso)

Para que todo funcione, necesitas **3 terminales** abiertas (o pestañas de terminal):

### Terminal 1: Base de Datos 🗄️
Esto arranca PostgreSQL en segundo plano.
```powershell
docker-compose up -d
```
*(Si ya estaba corriendo, no hace falta hacerlo de nuevo, pero no hace daño).*

### Terminal 2: Backend (API) 🧠
Este es el servicio que hace las predicciones.
```powershell
# 1. Activar el entorno virtual (si no sale (.venv) al principio de la línea)
.\.venv\Scripts\Activate

# 2. Iniciar el servidor
python -m uvicorn backend.api.app.main:app --reload
```
*Deberías ver: `Uvicorn running on http://127.0.0.1:8000`*

### Terminal 3: Frontend (Web) 💻
Esta es la interfaz visual.
```powershell
# 1. Entrar en la carpeta frontend
cd frontend

# 2. Iniciar la web
npm run dev
```
*Deberías ver: `Local: http://localhost:5173/`*

---

## 🌐 Usar la Aplicación

1.  Abre tu navegador (Chrome/Edge).
2.  Ve a: **[http://localhost:5173](http://localhost:5173)**
3.  Verás el indicador **API: Online** en verde arriba a la derecha.
4.  Escribe algo como *"I am so happy today!"* y pulsa **Analyze**.

---

## ❓ Solución de Problemas Comunes

-   **Error "uvicorn no se reconoce"**: Asegúrate de haber ejecutado `.\.venv\Scripts\Activate` primero.
-   **Error en Frontend "vite no se reconoce"**: Asegúrate de estar dentro de la carpeta `frontend` (`cd frontend`) y haber ejecutado `npm install` alguna vez.
-   **API Offline (Rojo)**: Comprueba la Terminal 2. Si se cerró o dio error, la web no puede predecir nada.
