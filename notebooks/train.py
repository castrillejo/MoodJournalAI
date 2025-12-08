import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Rutas - Usamos rutas relativas desde "notebooks/"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetuning")
# Carpeta raíz del modelo (donde está base/)
MODEL_ROOT_DIR = os.path.join(BASE_DIR, "model-training", "download-model", "roberta-base-english")
MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "base") # El modelo base original

# Donde se guardarán los checkpoints (temporales)
OUTPUT_DIR = os.path.join(BASE_DIR, "model-training", "models", "checkpoints")
# Donde se guardará el modelo FINAL (junto a base/)
FINAL_MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "finetuned-emotion")
LOG_DIR = os.path.join(BASE_DIR, "model-training", "logs")

# Parámetros del Modelo
NUM_LABELS = 6
LABEL_MAP = {
    'joy': 0, 'sadness': 1, 'fear': 2, 
    'anger': 3, 'love': 4, 'surprise': 5
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL2ID = LABEL_MAP

# ==========================================
# FUNCIONES
# ==========================================

def compute_metrics(eval_pred):
    """Calcula métricas para evaluación durante el entrenamiento"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print("🚀 Iniciando configuración de Fine-Tuning...")

    # 1. Comprobaciones básicas
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ No se encuentra la carpeta de datos: {DATA_DIR}. Ejecuta prepare_dataset.py primero.")
    
    print(f"📂 Cargando modelo base desde: {MODEL_DIR}")
    
    # 2. Cargar Tokenizer y Modelo
    try:
        # Cargamos tokenizer desde la raíz del modelo (donde suelen estar los archivos json)
        # o desde base si están ahí. Asumimos estructura estándar:
        # roberta-base-english/ (tokenizer files)
        # roberta-base-english/base/ (safetensors)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    except OSError as e:
        print(f"❌ Error cargando modelo. Asegúrate de haber ejecutado download_roberta.py.\nDetalle: {e}")
        return

    print("✅ Modelo y Tokenizer cargados.")

    # 3. Cargar Datasets
    data_files = {
        'train': os.path.join(DATA_DIR, 'train.csv'),
        'validation': os.path.join(DATA_DIR, 'val.csv')
    }
    
    dataset = load_dataset('csv', data_files=data_files)
    print(f"📚 Datos cargados: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    # 4. Tokenización
    def tokenize_function(examples):
        return tokenizer(
            examples['texto_diario'], 
            truncation=True, 
            max_length=128, 
            padding=False # El padding dinámico lo hará el DataCollator
        )

    print("⚙️ Tokenizando datos...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Data Collator para padding dinámico (más eficiente)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. Configurar Argumentos de Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Hyperparámetros
        learning_rate=2e-5,
        per_device_train_batch_size=16, # Ajustar a 8 si da error de memoria en GPU
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        
        # Evaluación y Guardado
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        
        # Logging
        logging_dir=LOG_DIR,
        logging_steps=50,
        report_to=["tensorboard"],
        
        # Optimización
        fp16=torch.cuda.is_available(), # Usar hardware mixto si hay GPU
        dataloader_num_workers=0,      # Windows suele dar problemas con workers > 0
    )

    # 6. Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🏋️‍♂️ Todo listo. Para iniciar el entrenamiento, ejecuta este script.")
    print(f"   Los checkpoints irán a: {OUTPUT_DIR}")
    print(f"   El modelo FINAL irá a:  {FINAL_MODEL_DIR}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()
    
    # Confirmación de usuario para arrancar
    resp = input("\n¿Quieres comenzar el entrenamiento AHORA? (s/n): ")
    if resp.lower() == 's':
        print("🚀 Entrenando...")
        trainer.train()
        
        print(f"💾 Guardando modelo final en: {FINAL_MODEL_DIR}...")
        trainer.save_model(FINAL_MODEL_DIR)
        print("✅ Guardado exitoso.")
    else:
        print("🛑 Entrenamiento cancelado.")
