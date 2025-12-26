# 🔍 FASE 3: Visualización de Attention Weights - COMPLETADA ✅

## 🎯 ¿Qué se implementó?

La **Visualización de Attention Weights** permite ver qué palabras del texto influyeron más en la predicción del modelo RoBERTa. Es una técnica de **Explainable AI (XAI)** que hace el modelo más transparente y comprensible.

---

## 🧠 ¿Cómo funciona?

### **Concepto:**
Cuando RoBERTa lee un texto como *"I feel very happy today"*, presta más **atención** a ciertas palabras:
- **"happy"** → 95% atención (palabra clave para JOY)
- **"very"** → 15% atención (amplifica)
- **"feel"** → 10% atención
- **"I", "today"** → 5% (menos relevantes)

### **Técnicamente:**
Extraemos los **attention weights** de la última capa del transformer RoBERTa, específicamente la atención del token `[CLS]` (que se usa para clasificación) hacia todos los demás tokens.

---

## 🎨 Funcionalidades implementadas:

### 1. **Texto Resaltado** 
- Palabras coloreadas según su importancia
- Intensidad del color = nivel de influencia
- Font weight variable (más grueso = más importante)

### 2. **Gráfico de Barras de Attention**
- Top 10 palabras más influyentes
- Escala de colores Viridis
- Scores normalizados 0-100%

### 3. **Top 5 Palabras Clave**
- Lista ordenada de las 5 palabras más importantes
- Porcentaje de atención para cada una

### 4. **Toggle Interactivo**
- Checkbox "🔍 Mostrar Attention"
- Permite alternar entre modo normal y modo attention
- Sin necesidad de recargar la página

---

## 📡 Arquitectura

### **Backend (FastAPI):**

#### Nuevo endpoint:
```
POST /api/predict/attention
```

**Request:**
```json
{
  "text": "I feel so happy today!"
}
```

**Response:**
```json
{
  "predicted_emotion": "joy",
  "confidence": 0.94,
  "all_scores": [...],
  "attention": {
    "tokens": [" I", " feel", " so", " happy", " today", "!"],
    "scores": [0.15, 0.22, 0.18, 0.95, 0.12, 0.08]
  }
}
```

#### Modificaciones en código:

**`ml_service.py`:**
- Nuevo método `predict_with_attention()`
- Extrae attention weights con `output_attentions=True`
- Promedia sobre todos los attention heads
- Normaliza scores (0-1)
- Limpia tokens especiales de RoBERTa (`<s>`, `</s>`, `Ġ`)

**`routes/predict.py`:**
- Nuevo endpoint `/predict/attention`
- Documentación actualizada
- Retorna estructura con datos de attention

### **Frontend (Streamlit):**

**`streamlit_app.py`:**
- Nueva función `predict_emotion_with_attention()`
- Nueva función `create_attention_bar_chart()`
- Nueva función `render_attention_text()`
- Toggle checkbox para activar modo attention
- Sección de visualización de attention
- Gráficos interactivos con Plotly

---

## 🚀 Cómo usar:

### 1. Asegúrate de que ambos servidores estén corriendo:

**Terminal 1 (FastAPI):**
```powershell
uvicorn backend.api.app.main:app --reload
```

**Terminal 2 (Streamlit):**
```powershell
streamlit run backend\web\streamlit_app.py
```

### 2. En la interfaz de Streamlit:

1. Escribe un texto en inglés
2. **Activa el checkbox "🔍 Mostrar Attention"**
3. Click en "🧠 Analizar con Attention"
4. Observa:
   - Texto resaltado con palabras clave
   - Gráfico de barras Top 10
   - Lista Top 5 palabras
   - Distribución de emociones

---

## 📊 Ejemplo de Visualización

### Texto de entrada:
```
"I am extremely excited and happy about this amazing project!"
```

### Resultado esperado:

**Emoción:** JOY (92%)

**Top 5 Palabras:**
1. `excited` - 98%
2. `happy` - 95%
3. `amazing` - 87%
4. `extremely` - 45%
5. `project` - 23%

**Visualización:**
- Palabras "excited", "happy", "amazing" aparecerán con fondo dorado intenso
- Palabras menos relevantes ("I", "am", "and") con fondo claro

---

## 🎓 Aplicaciones de Attention Visualization:

### **Académicas:**
- Explicar cómo funciona el modelo
- Debugging de predicciones incorrectas
- Análisis de sesgos del modelo

### **Profesionales:**
- Transparencia para usuarios finales
- Compliance con regulaciones de IA explicable
- Confianza en el sistema

### **Investigación:**
- Análisis de qué patrones aprende RoBERTa
- Comparación de diferentes modelos
- Estudio de transferencia de conocimiento

---

## 🔧 Archivos Modificados:

```
✅ backend/api/app/ml_service.py
   - Agregado: predict_with_attention()
   
✅ backend/api/app/routes/predict.py
   - Agregado: POST /api/predict/attention
   
✅ backend/web/streamlit_app.py
   - Agregado: predict_emotion_with_attention()
   - Agregado: create_attention_bar_chart()
   - Agregado: render_attention_text()
   - Modificado: UI con toggle y visualización
```

---

## 🎯 Próximos pasos (Opcional):

### **Mejoras posibles:**
- [ ] Heatmap 2D de attention entre todos los tokens
- [ ] Comparación de attention entre diferentes capas
- [ ] Export de visualización como imagen
- [ ] Modo batch (múltiples textos a la vez)
- [ ] Attention para diferentes heads (no solo promedio)

### **FASE 4: Frontend React**
- Crear interfaz profesional con React
- Visualización más sofisticada con D3.js
- Animaciones de transición
- PWA para instalación

---

## 📚 Referencias Técnicas:

- **Attention Mechanism:** Vaswani et al., "Attention is All You Need" (2017)
- **RoBERTa:** Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
- **XAI:** "Explainable AI: Interpreting, Explaining and Visualizing Deep Learning" (2019)

---

**✅ FASE 3 COMPLETADA**

Fecha: 26/12/2025  
Modelo: RoBERTa-base fine-tuned  
Técnica: Attention Weight Visualization  
Framework: FastAPI + Streamlit + Plotly
