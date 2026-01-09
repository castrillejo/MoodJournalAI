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


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "finetuning")
MODEL_ROOT_DIR = os.path.join(BASE_DIR, "model-training", "download-model", "roberta-base-english")
MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "base")
OUTPUT_DIR = os.path.join(BASE_DIR, "model-training", "models", "checkpoints")
FINAL_MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "finetuned-emotion")
LOG_DIR = os.path.join(BASE_DIR, "model-training", "logs")

NUM_LABELS = 6
LABEL_MAP = {
    'joy': 0, 'sadness': 1, 'fear': 2, 
    'anger': 3, 'love': 4, 'surprise': 5
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL2ID = LABEL_MAP

def compute_metrics(eval_pred):
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
    print("Iniciando configuración de Fine-Tuning...")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"No se encuentra la carpeta de datos: {DATA_DIR}. Ejecuta prepare_dataset.py primero.")
    
    print(f"📂 Cargando modelo base desde: {MODEL_DIR}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, 
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
    except OSError as e:
        print(f"Error cargando modelo. Asegúrate de haber ejecutado download_roberta.py.\nDetalle: {e}")
        return

    print("Modelo y Tokenizer cargados.")

    data_files = {
        'train': os.path.join(DATA_DIR, 'train.csv'),
        'validation': os.path.join(DATA_DIR, 'val.csv')
    }
    
    dataset = load_dataset('csv', data_files=data_files)
    print(f"Datos cargados: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    def tokenize_function(examples):
        return tokenizer(
            examples['texto_diario'], 
            truncation=True, 
            max_length=128, 
            padding=False 
        )

    print("Tokenizando datos...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        logging_dir=LOG_DIR,
        logging_steps=50,
        report_to=["tensorboard"],
        
        fp16=torch.cuda.is_available(), 
        dataloader_num_workers=0,      
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\nTodo listo. Para iniciar el entrenamiento, ejecuta este script.")
    print(f"   Los checkpoints irán a: {OUTPUT_DIR}")
    print(f"   El modelo FINAL irá a:  {FINAL_MODEL_DIR}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()

    resp = input("\n¿Quieres comenzar el entrenamiento AHORA? (s/n): ")
    if resp.lower() == 's':
        print("Entrenando...")
        trainer.train()
        
        print(f"Guardando modelo final en: {FINAL_MODEL_DIR}...")
        trainer.save_model(FINAL_MODEL_DIR)
        print("Guardado exitoso.")
    else:
        print("Entrenamiento cancelado.")
