# 📥 Descarga de Modelo BETO

Esta carpeta contiene el script para descargar el modelo **BETO** (BERT Español) pre-entrenado desde Hugging Face Hub.

## 🎯 Modelo

- **Nombre:** `dccuchile/bert-base-spanish-wwm-cased`
- **Tipo:** BERT base entrenado en español
- **Parámetros:** ~110 millones
- **Corpus:** Wikipedia español, libros, noticias
- **Tamaño descarga:** ~420 MB

## 🚀 Uso

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar descarga

```bash
python download_beto.py
```

El script:
- ✅ Descarga el modelo y tokenizer desde Hugging Face
- ✅ Guarda todo en la carpeta `bert-base-spanish/`
- ✅ Verifica que la descarga fue exitosa
- ✅ Muestra información sobre cómo usar el modelo

## 📂 Estructura después de la descarga

```
download-model/
├── README.md
├── requirements.txt
├── download_beto.py
└── bert-base-spanish/          # ← Creada automáticamente
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.txt
    ├── special_tokens_map.json
    └── base/
        └── pytorch_model.bin   # ~420 MB
```

## 💡 Próximos pasos

Una vez descargado el modelo, puedes:

1. **Usarlo directamente para inferencia**
2. **Hacer fine-tuning** para análisis de sentimientos
3. **Experimentar** en notebooks

## 🔍 Verificar la descarga

El script automáticamente verifica que el modelo se descargó correctamente. Si ves el mensaje "✅ Modelo verificado correctamente", todo está listo.

## ⚠️ Notas

- **Conexión a internet:** Necesaria solo para la primera descarga
- **Espacio en disco:** ~500 MB libres recomendados
- **Tiempo estimado:** 2-5 minutos (dependiendo de tu conexión)
