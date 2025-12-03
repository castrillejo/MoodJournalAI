# 📥 Descarga de Modelo RoBERTa-base (Inglés)

Esta carpeta contiene el script para descargar el modelo **RoBERTa-base** (inglés) pre-entrenado desde Hugging Face Hub.

## 🎯 Modelo

- **Nombre:** `roberta-base`
- **Tipo:** RoBERTa base entrenado en inglés
- **Parámetros:** ~125 millones
- **Corpus:** BookCorpus, Wikipedia inglés, CC-News, OpenWebText, STORIES
- **Tamaño descarga:** ~500 MB
- **Arquitectura:** 12 capas, 768 dimensiones, 12 attention heads

## ✨ ¿Por qué RoBERTa en lugar de BERT?

- **Mejor entrenamiento:** Más datos, más tiempo, lotes más grandes
- **Sin NSP:** Eliminada la tarea de "Next Sentence Prediction"
- **Dynamic masking:** Patrones de masking cambian en cada época
- **Rendimiento:** Supera a BERT original en la mayoría de benchmarks

## 🚀 Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar descarga

```bash
python download_roberta.py
```

El script:
- ✅ Descarga el modelo y tokenizer desde Hugging Face
- ✅ Guarda todo en la carpeta `roberta-base-english/`
- ✅ Verifica que la descarga fue exitosa
- ✅ Muestra información sobre cómo usar el modelo

## 📂 Estructura después de la descarga

```
download-model/
├── README.md
├── requirements.txt
├── download_roberta.py
└── roberta-base-english/       # ← Creada automáticamente
    ├── config.json
    ├── tokenizer.json
    ├── vocab.json
    ├── merges.txt
    └── base/
        └── pytorch_model.bin   # ~500 MB
```

## 💡 Próximos pasos

Una vez descargado el modelo, puedes:

1. **Usarlo directamente para inferencia**
2. **Hacer fine-tuning** para análisis de sentimientos con tus 6 emociones
3. **Experimentar** en notebooks

## 🔍 Verificar la descarga

El script automáticamente verifica que el modelo se descargó correctamente. Si ves el mensaje "✅ Modelo verificado correctamente", todo está listo.

## 🎯 Para tu proyecto MoodJournalAI

Este modelo es perfecto para:
- Textos en **inglés** (tus datos en `entradas.csv`)
- Fine-tuning para **6 emociones**: joy, sadness, fear, anger, love, surprise
- Balance entre **rendimiento** y **tamaño**

## ⚠️ Notas

- **Conexión a internet:** Necesaria solo para la primera descarga
- **Espacio en disco:** ~500 MB libres recomendados
- **Tiempo estimado:** 3-7 minutos (dependiendo de tu conexión)
- **Idioma:** Optimizado para **inglés** (perfecto para tus datos)
