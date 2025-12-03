# Plan de Fine-Tuning para MoodJournalAI

## ✅ Estado Actual (lo que ya tienes)

### Datos
- **6,124 entradas** etiquetadas en `data/entradas.csv`
- **6 emociones:** joy, sadness, fear, anger, love, surprise
- **Textos en inglés** (perfecto para RoBERTa)

### Modelo
- **RoBERTa-base descargado:** `model-training/download-model/roberta-base-english/`
- ~500 MB en disco
- 125 millones de parámetros
- Listo para fine-tuning

### Infraestructura
- PostgreSQL funcionando
- Estructura de carpetas organizada
- Entorno virtual Python configurado

---

## 🧠 Fundamentos Teóricos del Fine-tuning

### ¿Qué son los Embeddings?

Los **embeddings** son representaciones numéricas de texto que capturan su significado. RoBERTa convierte cada palabra/frase en un vector de **768 números**.

#### Ejemplo simplificado (4 dimensiones):

```
Texto: "I feel happy today"

Tokens → Embeddings:
"I"      → [0.1,  0.2,  0.1,  0.3]
"feel"   → [0.3,  0.8,  0.2,  0.1]
"happy"  → [0.9, -0.1,  0.7,  0.2]
"today"  → [0.2,  0.3,  0.1,  0.4]

Embedding combinado: [0.38, 0.30, 0.28, 0.25]
```

**Cada dimensión captura un aspecto del significado:**
- Dimensión 0: ¿Es positivo o negativo?
- Dimensión 1: ¿Es emocional?
- Dimensión 2: ¿Es activo o pasivo?
- Dimensión 3: ¿Es sobre el presente?

### ¿Cómo funciona el Classifier?

El **classifier head** es una capa que multiplica el embedding por pesos aprendidos:

```python
Embedding:      [0.38, 0.30, 0.28, 0.25]

Pesos para JOY:      [+2.0, +1.5, +1.0, +0.5]
Pesos para SADNESS:  [-2.0, +1.5, -0.5, +0.5]
...

Score JOY = (0.38 × 2.0) + (0.30 × 1.5) + (0.28 × 1.0) + (0.25 × 0.5)
          = 0.76 + 0.45 + 0.28 + 0.125
          = 1.615 ✅ GANADOR

Score SADNESS = (0.38 × -2.0) + (0.30 × 1.5) + (0.28 × -0.5) + (0.25 × 0.5)
              = -0.76 + 0.45 - 0.14 + 0.125
              = -0.325

Predicción: JOY (82% de confianza)
```

### ¿Qué cambia durante el Fine-tuning?

#### ANTES del fine-tuning:
- **Embeddings:** Optimizados para inglés general (Wikipedia, libros)
- **Classifier:** No existe o tiene pesos aleatorios
- **Resultado:** No puede clasificar emociones

#### DURANTE el fine-tuning (con tus 6,124 entradas):
1. El modelo lee "I feel happy" → JOY
2. Genera embedding: `[0.23, -0.45, 0.67, ...]`
3. Classifier predice: JOY (25%) ← Baja confianza
4. **Ajusta pesos:** "Cuando veo 'happy', aumentar dimensión X, reducir dimensión Y"
5. Repite 6,124 veces × 3 epochs = 18,372 ajustes

#### DESPUÉS del fine-tuning:
- **Embeddings:** Optimizados para emociones en diarios
- **Classifier:** Pesos entrenados para 6 emociones específicas
- **Resultado:** Predice JOY (94% de confianza) ✅

---

## 🚀 Plan de Acción (4 Fases)

### Fase 1: Preprocesamiento de Datos (1 día)

**Objetivo:** Convertir `entradas.csv` en datasets train/val/test

#### Tareas:
1. ✅ Crear script `prepare_dataset.py`
2. ✅ Cargar `entradas.csv` (6,124 entradas)
3. ✅ Limpiar textos (remover NaN, textos vacíos)
4. ✅ Mapear emociones a números:
   ```python
   emotion_map = {
       'joy': 0,
       'sadness': 1,
       'fear': 2,
       'anger': 3,
       'love': 4,
       'surprise': 5
   }
   ```
5. ✅ Dividir: 80% train / 10% val / 10% test
6. ✅ Guardar CSVs en `model-training/data/`

#### Output esperado:
```
model-training/data/
├── train.csv      (~4,900 entradas)
├── val.csv        (~610 entradas)
└── test.csv       (~614 entradas)
```

#### Script básico:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv("../../data/entradas.csv")

# Seleccionar columnas
df = df[['texto_diario', 'emocion_principal']].dropna()

# Mapear emociones
emotion_map = {'joy': 0, 'sadness': 1, 'fear': 2, 
               'anger': 3, 'love': 4, 'surprise': 5}
df['label'] = df['emocion_principal'].map(emotion_map)

# Dividir
train_df, temp_df = train_test_split(df, test_size=0.2, 
                                     stratify=df['label'], 
                                     random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                   stratify=temp_df['label'], 
                                   random_state=42)

# Guardar
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
```

---

### Fase 2: Configuración del Fine-tuning (1 día)

**Objetivo:** Configurar el entrenamiento de RoBERTa

#### Tareas:
1. ✅ Crear script `train.py`
2. ✅ Cargar RoBERTa-base desde local
3. ✅ Añadir classifier head (6 clases)
4. ✅ Configurar hiperparámetros
5. ✅ Configurar logging (TensorBoard)
6. ✅ Configurar guardado de checkpoints

#### Script de entrenamiento:
```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 1. Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(
    "../download-model/roberta-base-english"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "../download-model/roberta-base-english/base",
    num_labels=6,
    id2label={0: 'joy', 1: 'sadness', 2: 'fear', 
              3: 'anger', 4: 'love', 5: 'surprise'},
    label2id={'joy': 0, 'sadness': 1, 'fear': 2, 
              'anger': 3, 'love': 4, 'surprise': 5}
)

# 2. Cargar datos
dataset = load_dataset('csv', data_files={
    'train': 'data/train.csv',
    'validation': 'data/val.csv'
})

# 3. Tokenizar
def tokenize_function(examples):
    return tokenizer(examples['texto_diario'], 
                    truncation=True, 
                    padding='max_length', 
                    max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./models/roberta-sentiment-6emotions",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=50,
)

# 5. Entrenar
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

trainer.train()

# 6. Guardar modelo final
model.save_pretrained("./models/final")
tokenizer.save_pretrained("./models/final")
```

---

### Fase 3: Entrenamiento (2-4 horas con GPU)

**Objetivo:** Entrenar RoBERTa para clasificar 6 emociones

#### Hardware recomendado:
- **GPU (recomendado):** Google Colab (gratis), Kaggle, o local
  - Tiempo: 2-4 horas
- **CPU (lento):** Posible pero tardará 1-2 días

#### Proceso:
```bash
# Activar entorno virtual
.\.venv\Scripts\Activate

# Ir a la carpeta de scripts
cd model-training/scripts

# Ejecutar entrenamiento
python train.py
```

#### Output durante entrenamiento:
```
Epoch 1/3
  Step 100/306: Loss=1.456, Learning Rate=2e-5
  Step 200/306: Loss=1.123, Learning Rate=2e-5
  Step 306/306: Loss=0.892
  Evaluation: F1=0.65, Accuracy=0.63

Epoch 2/3
  Step 100/306: Loss=0.745, Learning Rate=2e-5
  Step 200/306: Loss=0.623, Learning Rate=2e-5
  Step 306/306: Loss=0.521
  Evaluation: F1=0.74, Accuracy=0.72

Epoch 3/3
  Step 100/306: Loss=0.456, Learning Rate=2e-5
  Step 200/306: Loss=0.389, Learning Rate=2e-5
  Step 306/306: Loss=0.312
  Evaluation: F1=0.79, Accuracy=0.77

Training complete! Best model saved to ./models/final
```

#### Archivos generados:
```
model-training/
├── models/
│   ├── roberta-sentiment-6emotions/  # Checkpoints
│   │   ├── checkpoint-306/
│   │   ├── checkpoint-612/
│   │   └── checkpoint-918/
│   │
│   └── final/                        # Mejor modelo
│       ├── config.json
│       ├── model.safetensors (~500 MB)
│       ├── vocab.json
│       └── merges.txt
│
└── logs/
    └── events.out.tfevents...        # Para TensorBoard
```

---

### Fase 4: Evaluación (1 hora)

**Objetivo:** Verificar calidad del modelo en datos no vistos

#### Script de evaluación:
```python
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Cargar modelo entrenado
classifier = pipeline(
    "text-classification",
    model="./models/final",
    tokenizer="./models/final"
)

# 2. Cargar test set
test_df = pd.read_csv("data/test.csv")

# 3. Hacer predicciones
predictions = []
for text in test_df['texto_diario']:
    pred = classifier(text)[0]
    predictions.append(pred['label'])

# 4. Métricas
print(classification_report(
    test_df['emocion_principal'], 
    predictions
))

# 5. Matriz de confusión
cm = confusion_matrix(
    test_df['emocion_principal'], 
    predictions,
    labels=['joy', 'sadness', 'fear', 'anger', 'love', 'surprise']
)
print(cm)
```

#### Resultados esperados:
```
              precision    recall  f1-score   support

         joy       0.85      0.82      0.83       120
     sadness       0.78      0.81      0.79       115
        fear       0.72      0.68      0.70        95
       anger       0.75      0.79      0.77       110
        love       0.81      0.84      0.82       100
    surprise       0.69      0.65      0.67        74

    accuracy                           0.77       614
   macro avg       0.77      0.77      0.76       614
weighted avg       0.77      0.77      0.77       614
```

**Meta de éxito:** F1-Score > 0.70 (70% de precisión)

---

## 📊 Análisis de Datos Recomendado

Antes del fine-tuning, es importante analizar la distribución de emociones:

```python
import pandas as pd

df = pd.read_csv("../data/entradas.csv")
print(df['emocion_principal'].value_counts())
```

**Distribución ideal:** 800-1500 ejemplos por emoción

**Si hay desbalance:**
- Usar `class_weight` en el Trainer
- Data augmentation (parafrasear textos)
- Oversampling de clases minoritarias

---

## ⚙️ Hiperparámetros Explicados

```python
TrainingArguments(
    # Épocas: cuántas veces el modelo ve todos los datos
    num_train_epochs=3,          # 3 pasadas completas
    
    # Batch sizes: cuántos ejemplos procesa a la vez
    per_device_train_batch_size=16,   # GPU: 16, CPU: 8
    per_device_eval_batch_size=32,    # Más grande en eval
    
    # Learning rate: cuánto ajusta los pesos en cada paso
    learning_rate=2e-5,          # 0.00002 (común para BERT/RoBERTa)
    
    # Weight decay: regularización para evitar overfitting
    weight_decay=0.01,
    
    # Evaluación: cuándo evaluar el modelo
    evaluation_strategy="epoch",  # Al final de cada época
    
    # Guardado: cuándo guardar checkpoints
    save_strategy="epoch",
    
    # Mejor modelo: cargar el mejor al final
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # Usar F1-Score
)
```

---

## 🎯 Estructura de Archivos Final

```
MoodJournalAI/
├── data/
│   └── entradas.csv              # Datos originales (6,124)
│
└── model-training/
    ├── data/                     # Datos procesados
    │   ├── train.csv             # 4,900 entradas
    │   ├── val.csv               # 610 entradas
    │   └── test.csv              # 614 entradas
    │
    ├── scripts/                  # Scripts de trabajo
    │   ├── prepare_dataset.py    # Paso 1
    │   ├── train.py              # Paso 2 y 3
    │   ├── evaluate.py           # Paso 4
    │   └── predict.py            # Uso del modelo
    │
    ├── models/                   # Modelos entrenados
    │   ├── roberta-sentiment-6emotions/
    │   └── final/                # Mejor modelo
    │       ├── config.json
    │       └── model.safetensors
    │
    ├── logs/                     # Logs de entrenamiento
    │   └── tensorboard/
    │
    └── download-model/           # Modelo base original
        └── roberta-base-english/
```

---

## 🚨 Troubleshooting

### Error: Out of Memory (GPU)
```python
# Solución: reducir batch size
per_device_train_batch_size=8  # en vez de 16
```

### Error: Training muy lento (CPU)
```python
# Solución: usar Google Colab gratis con GPU
# O reducir datos de entrenamiento para pruebas
train_df_sample = train_df.sample(1000)
```

### Error: Overfitting (train accuracy alta, val accuracy baja)
```python
# Solución: añadir regularización
weight_decay=0.1  # aumentar de 0.01
# O reducir épocas
num_train_epochs=2
```

---

## 📈 Próximos Pasos Después del Fine-tuning

1. **Integrar con backend:** API FastAPI para predicciones
2. **Crear interfaz:** Frontend para probar el modelo
3. **Deployment:** Servir modelo en producción
4. **Monitoreo:** Tracking de predicciones y métricas

---

## 🎓 Recursos Adicionales

- **Hugging Face Transformers:** https://huggingface.co/docs/transformers/
- **RoBERTa paper:** https://arxiv.org/abs/1907.11692
- **Fine-tuning guide:** https://huggingface.co/docs/transformers/training

---

**Última actualización:** 2025-12-03  
**Modelo:** RoBERTa-base (inglés)  
**Dataset:** 6,124 entradas en inglés  
**Objetivo:** Clasificar 6 emociones (joy, sadness, fear, anger, love, surprise)
