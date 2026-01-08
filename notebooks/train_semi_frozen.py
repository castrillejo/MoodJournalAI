import os
import torch
import numpy as np
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
BASE_MODEL_DIR = os.path.join(MODEL_ROOT_DIR, "base")

OUTPUT_ROOT = os.path.join(BASE_DIR, "model-training", "models")
LOG_ROOT = os.path.join(BASE_DIR, "model-training", "logs")

NUM_LABELS = 6
LABEL_MAP = {
    "joy": 0, "sadness": 1, "fear": 2,
    "anger": 3, "love": 4, "surprise": 5
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
LABEL2ID = LABEL_MAP

MAX_LENGTH = 128
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
TRAIN_BS = 16
EVAL_BS = 64

UNFREEZE_VARIANTS = [2, 4, 6]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    precision = precision_score(labels, preds, average="weighted")
    recall = recall_score(labels, preds, average="weighted")

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def freeze_semi_frozen(model, unfreeze_last_n_layers: int):
    # Congela TODO roberta
    for p in model.roberta.parameters():
        p.requires_grad = False

    # Descongela últimas N capas del encoder
    layers = model.roberta.encoder.layer
    total_layers = len(layers)
    n = max(0, min(unfreeze_last_n_layers, total_layers))

    if n > 0:
        for layer in layers[total_layers - n:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Classifier siempre entrenable
    for p in model.classifier.parameters():
        p.requires_grad = True

def train_variant(unfreeze_n: int, tokenizer, tokenized_datasets, data_collator):
    variant_name = f"semi-frozen{unfreeze_n}"
    final_model_dir = os.path.join(MODEL_ROOT_DIR, variant_name)

    output_dir = os.path.join(OUTPUT_ROOT, f"checkpoints-{variant_name}")
    log_dir = os.path.join(LOG_ROOT, f"{variant_name}")

    print(f"\n🏋️ Entrenando {variant_name} (descongelando últimas {unfreeze_n} capas)...")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_DIR,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    freeze_semi_frozen(model, unfreeze_n)

    training_args = TrainingArguments(
        output_dir=output_dir,

        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS,
        per_device_eval_batch_size=EVAL_BS,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",

        logging_dir=log_dir,
        logging_steps=20,
        report_to=["tensorboard"],

        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(final_model_dir)

    print(f"✅ Guardado: {final_model_dir}")

def main():
    # 1) Cargar dataset (una vez)
    data_files = {
        "train": os.path.join(DATA_DIR, "train.csv"),
        "validation": os.path.join(DATA_DIR, "val.csv"),
    }
    dataset = load_dataset("csv", data_files=data_files)

    # 2) Tokenizer (una vez)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

    def tokenize_function(examples):
        return tokenizer(
            examples["texto_diario"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3) Entrenar variantes (2,4,6)
    for n in UNFREEZE_VARIANTS:
        train_variant(n, tokenizer, tokenized_datasets, data_collator)

    print("\n🎉 Entrenamiento de variantes semi-frozen completado.")

if __name__ == "__main__":
    main()