# 🚀 Guía de Migración al Portátil

Esta guía te ayudará a transferir el proyecto MoodJournalAI desde tu PC de casa a tu portátil usando un pendrive.

---

## 📦 PASO 1: Preparar en el PC de Casa

### Opción A: Transferencia Completa (Recomendado para la presentación)

**Copia TODA la carpeta del proyecto:**
```
C:\MoodJournalAI  →  Pendrive
```

**Tamaño aproximado:** ~7-10 GB (incluye todos los modelos entrenados)

**Contenido que se copiará:**
- ✅ Modelos entrenados (frozen, semi-frozen2/4/6, finetuned)
- ✅ Modelo RoBERTa base
- ✅ Código fuente (backend + frontend)
- ✅ Datos (entradas.csv, usuarios.csv)
- ✅ Configuraciones

**⚠️ IMPORTANTE: NO copies estas carpetas** (las regenerarás en el portátil):
- `.venv/` (entorno virtual - ocupa mucho y es específico del PC)
- `frontend/node_modules/` (dependencias de npm - se regeneran)
- `model-training/logs/` (opcional, solo si necesitas los históricos de entrenamiento)

### Cómo excluir carpetas al copiar:

```powershell
# Desde PowerShell en tu PC de casa
# 1. Crea una carpeta temporal
New-Item -Path "D:\MoodJournalAI_Transfer" -ItemType Directory

# 2. Copia excluyendo .venv y node_modules
robocopy "C:\MoodJournalAI" "D:\MoodJournalAI_Transfer" /E /XD ".venv" "node_modules" "__pycache__"

# 3. Ahora copia D:\MoodJournalAI_Transfer al pendrive
```

---

## 💻 PASO 2: Configurar en el Portátil

### Requisitos Previos en el Portátil

Asegúrate de tener instalado:

1. **Python 3.10+** → https://www.python.org/downloads/
2. **Node.js 18+** → https://nodejs.org/
3. **Docker Desktop** → https://www.docker.com/products/docker-desktop
4. **Git** (opcional) → https://git-scm.com/

---

## ⚙️ PASO 3: Setup del Proyecto

### 3.1 Copiar al Portátil

```powershell
# Copia la carpeta del pendrive al portátil
# Por ejemplo:
Copy-Item "E:\MoodJournalAI" -Destination "C:\MoodJournalAI" -Recurse
```

### 3.2 Crear Entorno Virtual de Python

Abre PowerShell en la carpeta del proyecto:

```powershell
# Navegar al proyecto
cd C:\MoodJournalAI

# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
.\.venv\Scripts\Activate

# Deberías ver (.venv) al inicio del prompt
```

### 3.3 Instalar Dependencias de Python

```powershell
# Asegúrate de que el entorno virtual está activado
# Instalar dependencias del backend
pip install -r backend/requirements.txt

# Si da error, prueba:
pip install --upgrade pip
pip install -r backend/requirements.txt
```

**Tiempo estimado:** 5-10 minutos (depende de la conexión a internet)

### 3.4 Instalar Dependencias del Frontend

```powershell
# Ir a la carpeta frontend
cd frontend

# Instalar dependencias de npm
npm install

# Volver a la raíz
cd ..
```

**Tiempo estimado:** 3-5 minutos

---

## 🐳 PASO 4: Iniciar Docker

```powershell
# Iniciar Docker Desktop desde el menú de Windows
# Espera a que el icono se ponga en verde

# Levantar la base de datos
docker-compose up -d

# Verificar que está corriendo
docker ps
# Deberías ver: moodjournal_postgres y moodjournal_etl
```

---

## ✅ PASO 5: Verificar que Todo Funciona

### Terminal 1: Backend
```powershell
# Activar entorno virtual si no está activado
.\.venv\Scripts\Activate

# Iniciar backend
python -m uvicorn backend.api.app.main:app --reload
```

**✅ Deberías ver:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     🚀 Iniciando MoodJournalAI API...
```

### Terminal 2: Frontend
```powershell
# En una nueva terminal
cd C:\MoodJournalAI\frontend

# Iniciar frontend
npm run dev
```

**✅ Deberías ver:**
```
VITE ready in XXX ms
➜  Local:   http://localhost:5173/
```

### Verificación Final

1. Abre tu navegador en **http://localhost:5173**
2. Verifica que aparece **"API: Online"** en verde
3. Prueba la sección "Compare Predictions" escribiendo un texto
4. Verifica que la "Model Overview" muestra las métricas

---

## 🔧 Solución de Problemas Comunes

### ❌ Error: "Python no se reconoce como comando"
**Solución:** Reinstala Python y marca "Add Python to PATH" durante la instalación

### ❌ Error: "npm no se reconoce como comando"
**Solución:** Reinstala Node.js y reinicia la terminal

### ❌ Error: "Cannot find module 'transformers'"
**Solución:**
```powershell
.\.venv\Scripts\Activate
pip install -r backend/requirements.txt
```

### ❌ Error: "Module not found" en el frontend
**Solución:**
```powershell
cd frontend
rm -r node_modules
rm package-lock.json
npm install
```

### ❌ Docker no arranca
**Solución:**
1. Abre Docker Desktop
2. Espera a que termine de iniciar (icono verde)
3. Vuelve a ejecutar `docker-compose up -d`

### ❌ El backend carga pero dice "Model not found"
**Solución:** Verifica que copiaste la carpeta `model-training/models/` completa

### ❌ "API: Offline" en el frontend
**Solución:**
1. Verifica que el backend está corriendo (Terminal 1)
2. Abre http://127.0.0.1:8000/docs para ver si la API responde
3. Revisa los logs de la terminal del backend

---

## 📋 Checklist Pre-Presentación

Antes de la presentación, verifica:

- [ ] ✅ Python 3.10+ instalado → `python --version`
- [ ] ✅ Node.js 18+ instalado → `node --version`
- [ ] ✅ Docker Desktop instalado y corriendo
- [ ] ✅ Proyecto copiado en `C:\MoodJournalAI`
- [ ] ✅ Entorno virtual creado (`.venv/`)
- [ ] ✅ Dependencias de Python instaladas
- [ ] ✅ Dependencias de npm instaladas
- [ ] ✅ Docker corriendo → `docker ps`
- [ ] ✅ Backend corriendo → http://127.0.0.1:8000/docs
- [ ] ✅ Frontend corriendo → http://localhost:5173
- [ ] ✅ API conectada (indicador "Online" en verde)
- [ ] ✅ Prueba de predicción funcionando

---

## 🎯 Resumen Rápido

**En el PC de casa:**
```powershell
# Copiar todo excepto .venv y node_modules
robocopy "C:\MoodJournalAI" "D:\MoodJournalAI_Transfer" /E /XD ".venv" "node_modules" "__pycache__"
# Copiar D:\MoodJournalAI_Transfer al pendrive
```

**En el portátil:**
```powershell
# 1. Copiar del pendrive
Copy-Item "E:\MoodJournalAI" -Destination "C:\MoodJournalAI" -Recurse

# 2. Crear entorno virtual
cd C:\MoodJournalAI
python -m venv .venv
.\.venv\Scripts\Activate

# 3. Instalar dependencias
pip install -r backend/requirements.txt
cd frontend
npm install
cd ..

# 4. Iniciar Docker
docker-compose up -d

# 5. Terminal 1: Backend
python -m uvicorn backend.api.app.main:app --reload

# 6. Terminal 2: Frontend
cd frontend
npm run dev

# 7. Abrir http://localhost:5173
```