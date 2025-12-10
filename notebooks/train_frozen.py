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

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetuning")
MODEL_ROOT_DIR = os.path.join(BASE_DIR, "model-training", "download-model", "roberta-base-english")
MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "base")

# Donde se guardarán los checkpoints (temporales)
OUTPUT_DIR = os.path.join(BASE_DIR, "model-training", "models", "checkpoints-frozen")
# Donde se guardará el modelo FINAL (junto a base/)
FINAL_MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "frozen-classifier")
LOG_DIR = os.path.join(BASE_DIR, "model-training", "logs-frozen")

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

def freeze_base_model(model):
    """
    Congela todas las capas de RoBERTa EXCEPTO el clasificador.
    Esto es Feature Extraction / Linear Probing.
    """
    print("🔒 Congelando capas base de RoBERTa...")
    
    # Contar parámetros antes
    total_params = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Congelar TODO el modelo base (embeddings + encoder)
    for param in model.roberta.parameters():
        param.requires_grad = False
    
    # El clasificador se mantiene descongelado (por defecto viene con requires_grad=True)
    # No necesitamos hacer nada, pero lo verificamos explícitamente:
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Contar después
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total_params - trainable_after
    
    print(f"  ✅ Parámetros totales:    {total_params:,}")
    print(f"  🔒 Parámetros congelados: {frozen:,} ({frozen/total_params*100:.1f}%)")
    print(f"  🎓 Parámetros entrenables: {trainable_after:,} ({trainable_after/total_params*100:.1f}%)")
    print(f"     (Solo la capa clasificadora)\n")

def main():
    print("🚀 Iniciando FEATURE EXTRACTION (capas congeladas)...\n")

    # 1. Comprobaciones básicas
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"❌ No se encuentra la carpeta de datos: {DATA_DIR}")
    
    print(f"📂 Cargando modelo base desde: {MODEL_DIR}")
    
    # 2. Cargar Tokenizer y Modelo
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    except OSError as e:
        print(f"❌ Error cargando modelo. Detalle: {e}")
        return

    print("✅ Modelo y Tokenizer cargados.\n")

    # 3. CONGELAR CAPAS BASE
    freeze_base_model(model)

    # 4. Cargar Datasets
    data_files = {
        'train': os.path.join(DATA_DIR, 'train.csv'),
        'validation': os.path.join(DATA_DIR, 'val.csv')
    }
    
    dataset = load_dataset('csv', data_files=data_files)
    print(f"📚 Datos cargados: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    # 5. Tokenización
    def tokenize_function(examples):
        return tokenizer(
            examples['texto_diario'], 
            truncation=True, 
            max_length=128, 
            padding=False
        )

    print("⚙️ Tokenizando datos...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Configurar Argumentos de Entrenamiento
    # Como solo entrenamos el clasificador, podemos usar:
    # - Learning rate un poco mayor
    # - Menos épocas (converge más rápido)
    # - Batch size mayor (menos memoria)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        # Hyperparámetros ajustados para feature extraction
        learning_rate=5e-4,  # Mayor que en fine-tuning (era 2e-5)
        per_device_train_batch_size=32,  # Mayor que en fine-tuning (era 16)
        per_device_eval_batch_size=64,
        num_train_epochs=5,  # Unas pocas épocas más para el clasificador
        weight_decay=0.01,
        
        # Evaluación y Guardado
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        
        # Logging
        logging_dir=LOG_DIR,
        logging_steps=20,
        report_to=["tensorboard"],
        
        # Optimización
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
    )

    # 7. Inicializar Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        processing_class=tokenizer,  # Usar processing_class en vez de tokenizer
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🏋️‍♂️ Todo listo para Feature Extraction.")
    print(f"   Los checkpoints irán a: {OUTPUT_DIR}")
    print(f"   El modelo FINAL irá a:  {FINAL_MODEL_DIR}")
    print("\n📝 NOTA: Este entrenamiento será MUCHO más rápido (~10-15 min)")
    print("         porque solo entrenamos la capa clasificadora.\n")
    
    return trainer

if __name__ == "__main__":
    trainer = main()
    
    # Confirmación de usuario para arrancar
    resp = input("\n¿Quieres comenzar el entrenamiento con capas CONGELADAS? (s/n): ")
    if resp.lower() == 's':
        print("🚀 Entrenando (solo clasificador)...")
        trainer.train()
        
        print(f"💾 Guardando modelo final en: {FINAL_MODEL_DIR}...")
        trainer.save_model(FINAL_MODEL_DIR)
        print("✅ Guardado exitoso.")
        print("\n" + "="*70)
        print("🎉 Entrenamiento completado!")
        print("="*70)
        print("\nAhora puedes comparar 3 modelos:")
        print("  1. BASE (clasificador aleatorio)")
        print("  2. FROZEN (solo clasificador entrenado) ← Este")
        print("  3. FINE-TUNED (todo entrenado)")
    else:
        print("🛑 Entrenamiento cancelado.")
