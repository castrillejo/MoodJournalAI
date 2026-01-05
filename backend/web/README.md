# MoodJournalAI - Streamlit Web App

Interfaz web interactiva para clasificación de emociones.

## 🚀 Inicio Rápido

### Paso 1: Asegúrate de que la API FastAPI esté corriendo

En una terminal:
```powershell
cd C:\MoodJournalAI
.\.venv\Scripts\Activate
uvicorn backend.api.app.main:app --reload
```

### Paso 2: Ejecutar la aplicación Streamlit

En **otra terminal**:
```powershell
cd C:\MoodJournalAI
.\.venv\Scripts\Activate
streamlit run backend\web\streamlit_app.py
```

### Paso 3: Usar la aplicación

1. Se abrirá automáticamente en tu navegador: http://localhost:8501
2. Escribe un texto en inglés
3. Click en "🔍 Analizar Emoción"
4. ¡Disfruta de los resultados visuales!

## 🎨 Características

### ✨ Interfaz Visual
- **Diseño moderno** con gradientes y colores personalizados por emoción
- **Emojis grandes** para identificar rápidamente la emoción
- **Gráficos interactivos** con Plotly:
  - Gráfico de barras horizontal (todas las emociones)
  - Gráfico de pastel (top 3 emociones)

### 🔧 Funcionalidades
- ✅ Análisis de emoción en tiempo real
- ✅ Ejemplos rápidos con un click
- ✅ Indicador de estado de la API
- ✅ Métricas de confianza
- ✅ Detalles completos en formato JSON

### 🎭 Emociones Detectadas
- 😊 Joy (Alegría) - Dorado
- 😢 Sadness (Tristeza) - Azul
- 😨 Fear (Miedo) - Morado
- 😡 Anger (Ira) - Rojo
- 💕 Love (Amor) - Rosa
- 😮 Surprise (Sorpresa) - Naranja

## 📸 Capturas

La aplicación incluye:
- Header con título y subtítulo
- Sidebar con información del proyecto
- Área de texto para input del usuario
- Botones de ejemplos rápidos
- Resultado principal con emoji gigante y color personalizado
- Métrica de confianza
- Dos gráficos interactivos (barras y pastel)
- Detalles técnicos expandibles

## ⚙️ Configuración

### Puerto personalizado
```powershell
streamlit run backend\web\streamlit_app.py --server.port 8502
```

### Modo oscuro/claro
Se puede cambiar desde el menú de Streamlit (esquina superior derecha)

## 🔧 Troubleshooting

### Error: "No se puede conectar a la API"
**Solución:** Asegúrate de que FastAPI esté corriendo:
```powershell
uvicorn backend.api.app.main:app --reload
```

### Puerto ocupado
**Solución:** Streamlit usará automáticamente el siguiente puerto disponible (8502, 8503, etc.)

### La página no carga
**Solución:** Verifica que tengas todas las dependencias:
```powershell
pip install streamlit plotly requests
```

## 💡 Consejos de Uso

1. **Primera predicción lenta**: El modelo se carga en la primera request (~2-5 segundos)
2. **Predicciones siguientes**: Muy rápidas (~50-200ms)
3. **Usa ejemplos**: Los botones de ejemplo te ayudan a probar rápidamente
4. **Explora gráficos**: Los gráficos de Plotly son interactivos (hover, zoom, etc.)

## 📊 Arquitectura

```
Usuario → Streamlit App (localhost:8501)
              ↓
         HTTP Request
              ↓
    FastAPI (localhost:8000)
              ↓
         RoBERTa Model
              ↓
         JSON Response
              ↓
    Gráficos Visuales
```

## 🎯 Próximos pasos

- [ ] Añadir historial de predicciones
- [ ] Guardar resultados en CSV
- [ ] Visualización de attention weights
- [ ] Modo batch (analizar múltiples textos)
- [ ] Comparación de textos

## 🌐 Deployment

Para compartir con otros (opcional):
```powershell
streamlit run backend\web\streamlit_app.py --server.headless true
```

O deploying en Streamlit Cloud (gratis): https://streamlit.io/cloud
