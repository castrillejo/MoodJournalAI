# 🏠 Guía de Setup Completo - PC de Casa (Windows + RTX 4060)

Esta guía te permite configurar el proyecto **MoodJournalAI** desde cero en tu PC de casa, replicando exactamente el entorno del portátil.

**Hardware objetivo:** PC Windows con NVIDIA RTX 4060 (8GB VRAM)

---

## 📋 Checklist de Instalación

- [ ] Git instalado
- [ ] Python 3.10+ instalado
- [ ] Docker Desktop instalado
- [ ] DBeaver instalado
- [ ] Repositorio clonado
- [ ] Entorno virtual creado
- [ ] Dependencias instaladas
- [ ] Docker contenedores levantados
- [ ] ETL ejecutado
- [ ] RoBERTa descargado
- [ ] CUDA configurado para RTX 4060

---

## 🚀 PARTE 1: Instalaciones Previas (30-60 minutos)

### 1.1 - Instalar Git para Windows

#### Descargar:
- **URL:** https://git-scm.com/download/win
- **Archivo:** `Git-2.43.0-64-bit.exe` (o versión más reciente)

#### Instalar:
1. Ejecutar el instalador
2. **Opciones recomendadas:**
   - Editor: "Use Visual Studio Code as Git's default editor" (si tienes VS Code)
   - PATH: "Git from the command line and also from 3rd-party software"
   - Line endings: "Checkout Windows-style, commit Unix-style"
   - Terminal: "Use Windows' default console window"
3. Siguiente → Siguiente → Install

#### Verificar:
```powershell
git --version
# Output esperado: git version 2.43.0.windows.1
```

---

### 1.2 - Instalar Python 3.10 o superior

#### Descargar:
- **URL:** https://www.python.org/downloads/
- **Versión:** Python 3.10.11 o 3.11.x (recomendado para PyTorch)

#### Instalar:
1. Ejecutar instalador
2. ✅ **IMPORTANTE:** Marcar "Add Python to PATH"
3. Clic en "Install Now"
4. Esperar instalación

#### Verificar:
```powershell
python --version
# Output esperado: Python 3.10.11 (o 3.11.x)

pip --version
# Output esperado: pip 23.x.x
```

---

### 1.3 - Instalar Docker Desktop

#### Descargar:
- **URL:** https://www.docker.com/products/docker-desktop/
- **Archivo:** `Docker Desktop Installer.exe`

#### Instalar:
1. Ejecutar instalador
2. Marcar "Use WSL 2 instead of Hyper-V" (recomendado)
3. Install → Esperar instalación
4. **Reiniciar el PC** cuando te lo pida

#### Configurar después del reinicio:
1. Abrir Docker Desktop
2. Aceptar términos y condiciones
3. Skip tutorial (opcional)
4. Dejar Docker Desktop ejecutándose en segundo plano

#### Verificar:
```powershell
docker --version
# Output esperado: Docker version 24.x.x

docker-compose --version
# Output esperado: Docker Compose version v2.x.x
```

---

### 1.4 - Instalar DBeaver (Cliente de Base de Datos)

#### Descargar:
- **URL:** https://dbeaver.io/download/
- **Versión:** DBeaver Community Edition (gratis)

#### Instalar:
1. Ejecutar instalador
2. Siguiente → Siguiente → Install
3. Finish

**Nota:** DBeaver lo configuraremos después de levantar PostgreSQL.

---

### 1.5 - Instalar Visual Studio Code (Opcional pero recomendado)

#### Descargar:
- **URL:** https://code.visualstudio.com/
- **Versión:** Windows 64-bit User Installer

#### Instalar:
1. Ejecutar instalador
2. Marcar "Add to PATH"
3. Marcar "Add 'Open with Code' to context menu"
4. Install

---

## 📂 PARTE 2: Clonar Repositorio y Configurar Proyecto (10 minutos)

### 2.1 - Crear estructura de directorios

```powershell
# Abrir PowerShell como Administrador (opcional) o normal

# Ir a C:\
cd C:\

# Crear carpeta del proyecto
mkdir MoodJournalAI
cd MoodJournalAI
```

---

### 2.2 - Clonar repositorio desde GitHub

```powershell
# Dentro de C:\MoodJournalAI

# Clonar repositorio (cambia la URL por tu repositorio)
git clone https://github.com/TU_USUARIO/MoodJournalAI.git .

# El punto (.) al final clona el contenido directamente en la carpeta actual
# Si prefieres crear subcarpeta, omite el punto
```

**Alternativa:** Si no has subido a GitHub aún:
```powershell
# Copiar manualmente desde tu portátil o usar GitHub Desktop
```

---

### 2.3 - Verificar estructura del proyecto

```powershell
# Ver estructura
dir

# Output esperado:
# - backend/
# - frontend/
# - data/
# - etl/
# - model-training/
# - docker/
# - docker-compose.yml
# - README.md
```

---

## 🐍 PARTE 3: Configurar Entorno Virtual Python (10 minutos)

### 3.1 - Crear entorno virtual

```powershell
# Asegúrate de estar en C:\MoodJournalAI
cd C:\MoodJournalAI

# Crear entorno virtual
python -m venv .venv
```

**Espera 1-2 minutos mientras se crea.**

---

### 3.2 - Activar entorno virtual

```powershell
# Activar entorno
.\.venv\Scripts\Activate

# Ahora verás (.venv) al inicio del prompt:
# (.venv) PS C:\MoodJournalAI>
```

**⚠️ IMPORTANTE:** Siempre que abras una nueva terminal PowerShell para este proyecto, debes ejecutar:
```powershell
cd C:\MoodJournalAI
.\.venv\Scripts\Activate
```

---

### 3.3 - Actualizar pip

```powershell
# Dentro del entorno virtual (.venv)
python -m pip install --upgrade pip
```

---

### 3.4 - Instalar dependencias base

```powershell
# Instalar dependencias para el proyecto
pip install pandas scikit-learn datasets accelerate
```

---

## 🎮 PARTE 4: Configurar PyTorch con CUDA para RTX 4060 (15 minutos)

### 4.1 - Verificar drivers NVIDIA

```powershell
# Verificar que tu RTX 4060 esté detectada
nvidia-smi
```

**Output esperado:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 546.01       Driver Version: 546.01       CUDA Version: 12.3   |
|-------------------------------+----------------------+----------------------+
|   0  NVIDIA GeForce RTX 4060  | 00000000:01:00.0  On |                  N/A |
```

**Si NO aparece:**
1. Descargar drivers desde: https://www.nvidia.com/Download/index.aspx
2. Seleccionar: RTX 4060 → Windows 11/10 → Descargar
3. Instalar y reiniciar PC

---

### 4.2 - Instalar PyTorch con CUDA 12.1

```powershell
# Asegúrate de tener el entorno virtual activado (.venv)

# Instalar PyTorch optimizado para RTX 4060
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Espera 5-10 minutos** (descarga ~2 GB)

---

### 4.3 - Verificar que PyTorch detecta la GPU

```powershell
# Ejecutar script de verificación
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

**Output esperado:**
```
CUDA disponible: True
GPU: NVIDIA GeForce RTX 4060
VRAM: 8.0 GB
```

✅ **Si ves esto, tu GPU está lista para entrenar.**

❌ **Si dice `False`:** Reinstala drivers NVIDIA o usa CUDA 11.8:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 4.4 - Instalar Transformers

```powershell
# Instalar Hugging Face Transformers
pip install transformers
```

---

## 🐳 PARTE 5: Levantar Docker y PostgreSQL (10 minutos)

### 5.1 - Verificar que Docker Desktop está ejecutándose

1. Buscar "Docker Desktop" en el menú inicio
2. Abrir Docker Desktop
3. Esperar a que diga "Engine running" (luz verde)

---

### 5.2 - Levantar contenedores

```powershell
# Asegúrate de estar en C:\MoodJournalAI
cd C:\MoodJournalAI

# Levantar PostgreSQL
docker-compose up -d
```

**Output esperado:**
```
[+] Running 2/2
 ✔ Network moodjournal_default      Created
 ✔ Container moodjournal_postgres   Started
```

---

### 5.3 - Verificar contenedores

```powershell
# Ver contenedores ejecutándose
docker ps

# Output esperado:
# CONTAINER ID   IMAGE         PORTS                    STATUS
# abc123...      postgres:15   0.0.0.0:5432->5432/tcp   Up 30 seconds
```

---

### 5.4 - Conectar DBeaver a PostgreSQL

1. Abrir **DBeaver**
2. Clic en "New Database Connection" (ícono de enchufe)
3. Seleccionar **PostgreSQL**
4. **Configuración:**
   - Host: `localhost`
   - Port: `5432`
   - Database: `moodjournal`
   - Username: `admin`
   - Password: `admin`
5. "Test Connection" → Descargar drivers si pide
6. "Finish"

✅ **Ahora puedes ver la base de datos en DBeaver**

---

## 📊 PARTE 6: Ejecutar ETL (Cargar Datos de Muestra) (5 minutos)

### 6.1 - Instalar dependencias del ETL

```powershell
# Ir a la carpeta etl
cd etl

# Instalar dependencias
pip install -r requirements.txt
```

---

### 6.2 - Ejecutar script de carga

```powershell
# Ejecutar ETL
python load_data.py
```

**Output esperado:**
```
Conectando a PostgreSQL...
✅ Conexión exitosa
Cargando usuarios...
✅ 100 usuarios cargados
Cargando entradas...
✅ 6,124 entradas cargadas
Proceso ETL completado
```

---

### 6.3 - Verificar datos en DBeaver

1. En DBeaver, expandir `moodjournal` → `Schemas` → `public` → `Tables`
2. Clic derecho en `entradas` → "View Data"
3. Deberías ver **6,124 filas**

✅ **Datos cargados correctamente**

---

## 🤖 PARTE 7: Descargar Modelo RoBERTa (10 minutos)

### 7.1 - Ir a carpeta de descarga

```powershell
# Volver a raíz
cd C:\MoodJournalAI

# Ir a carpeta de descarga de modelos
cd model-training\download-model
```

---

### 7.2 - Instalar dependencias

```powershell
# Instalar transformers si no lo hiciste antes
pip install -r requirements.txt
```

---

### 7.3 - Descargar RoBERTa-base

```powershell
# Ejecutar script de descarga
python download_roberta.py
```

**Espera 5-10 minutos** (descarga ~500 MB)

**Output esperado:**
```
============================================================
📥 DESCARGANDO MODELO ROBERTA-BASE (INGLÉS)
============================================================

📂 Guardando en: C:\MoodJournalAI\model-training\download-model\roberta-base-english

📚 Descargando tokenizer...
✅ Tokenizer descargado y guardado

🤖 Descargando modelo base...
✅ Modelo base descargado y guardado

✨ DESCARGA COMPLETADA
```

---

### 7.4 - Verificar descarga

```powershell
# Ver archivos descargados
dir roberta-base-english

# Output esperado:
# - vocab.json
# - merges.txt
# - tokenizer.json
# - base/
#   - model.safetensors (498 MB)
```

---

## ✅ PARTE 8: Verificación Final (5 minutos)

### 8.1 - Probar modelo descargado

```powershell
# Ir a carpeta notebooks
cd C:\MoodJournalAI\notebooks

# Ejecutar script de prueba
python test_sentiment.py
```

**Output esperado:**
```
======================================================================
🔍 PROBANDO CON ROBERTA-BASE (Tu modelo descargado)
======================================================================

📂 Cargando modelo local: ../model-training/download-model/roberta-base-english
✅ Modelo RoBERTa-base cargado

📝 Frase de prueba: "Today I feel very happy"

🎯 RESULTADOS:
  ├─ Dimensiones del embedding: torch.Size([1, 7, 768])
  ├─ Tokens procesados: 7
  └─ Vector por token: 768 dimensiones
```

✅ **Modelo funcionando correctamente**

---

### 8.2 - Checklist de verificación

Ejecuta cada comando y verifica el resultado:

```powershell
# 1. Python
python --version
# ✅ Python 3.10.x o superior

# 2. Git
git --version
# ✅ git version 2.x

# 3. Docker
docker --version
# ✅ Docker version 24.x

# 4. PyTorch con CUDA
python -c "import torch; print(torch.cuda.is_available())"
# ✅ True

# 5. GPU detectada
python -c "import torch; print(torch.cuda.get_device_name(0))"
# ✅ NVIDIA GeForce RTX 4060

# 6. PostgreSQL ejecutándose
docker ps
# ✅ moodjournal_postgres UP

# 7. Datos cargados
python -c "import pandas as pd; df = pd.read_csv('data/entradas.csv'); print(f'{len(df)} entradas')"
# ✅ 6124 entradas

# 8. Modelo descargado
dir model-training\download-model\roberta-base-english\base\model.safetensors
# ✅ Archivo existe (~498 MB)
```

---

## 🎯 PARTE 9: Preparar para Fine-tuning

### 9.1 - Crear carpetas necesarias

```powershell
cd C:\MoodJournalAI\model-training

# Crear carpetas si no existen
mkdir data
mkdir scripts
mkdir models
mkdir logs
```

---

### 9.2 - Estado final del proyecto

```
C:\MoodJournalAI\
├── .venv\                          ✅ Entorno virtual activo
├── data\
│   ├── entradas.csv                ✅ 6,124 entradas
│   └── usuarios.csv                ✅ 100 usuarios
├── etl\                            ✅ ETL ejecutado
├── model-training\
│   ├── data\                       📁 (para train/val/test)
│   ├── scripts\                    📁 (para prepare/train/evaluate)
│   ├── models\                     📁 (para modelos entrenados)
│   ├── logs\                       📁 (para TensorBoard)
│   └── download-model\
│       └── roberta-base-english\   ✅ Modelo descargado (500 MB)
├── notebooks\
│   └── test_sentiment.py           ✅ Probado y funcionando
└── docker-compose.yml              ✅ PostgreSQL ejecutándose
```

---

## 🚀 Próximos Pasos

Ahora que todo está configurado, puedes:

1. **Crear scripts de fine-tuning** (prepare_dataset.py, train.py, evaluate.py)
2. **Entrenar el modelo** con tu RTX 4060 (2-3 horas)
3. **Evaluar resultados** y obtener métricas
4. **Usar el modelo** para predicciones

---

## 🔧 Comandos Útiles

### Detener Docker
```powershell
docker-compose down
```

### Levantar Docker de nuevo
```powershell
docker-compose up -d
```

### Activar entorno virtual (cada vez que abras terminal)
```powershell
cd C:\MoodJournalAI
.\.venv\Scripts\Activate
```

### Ver logs de Docker
```powershell
docker-compose logs -f
```

### Conectarse a PostgreSQL desde terminal
```powershell
docker exec -it moodjournal_postgres psql -U admin -d moodjournal
```

---

## ⚠️ Troubleshooting

### Docker no inicia
- Reiniciar Docker Desktop
- Verificar WSL 2: `wsl --status`

### GPU no detectada
- Actualizar drivers NVIDIA
- Reinstalar PyTorch con CUDA

### Puerto 5432 ocupado
- Otro PostgreSQL local ejecutándose
- Cambiar puerto en `docker-compose.yml`

---

## 📞 Soporte

Si algo falla, verifica paso a paso y revisa los logs.

---

**Última actualización:** 2025-12-03  
**Hardware:** PC Windows con RTX 4060 (8GB VRAM)  
**Python:** 3.10+  
**CUDA:** 12.1  
**PyTorch:** 2.1+
