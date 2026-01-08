import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from transformers import pipeline, AutoConfig
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

PROJECT_ROOT = Path(r"C:\MoodJournalAI")

TEST_CSV = PROJECT_ROOT / "data" / "finetuning" / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "api" / "app" / "assets" / "evaluation"

MODEL_ROOT_DIR = PROJECT_ROOT / "model-training" / "download-model" / "roberta-base-english"

# Tokenizer común (lo tienes completo aquí)
TOKENIZER_DIR = MODEL_ROOT_DIR / "base"

MODELS = {
    # frozen real = frozen-classifier (encoder congelado + head entrenado)
    "frozen": MODEL_ROOT_DIR / "frozen-classifier",
    "finetuned": MODEL_ROOT_DIR / "finetuned-emotion",
}

SEMI_FROZEN_VARIANTS = {
    "semi_frozen2": MODEL_ROOT_DIR / "semi-frozen2",
    "semi_frozen4": MODEL_ROOT_DIR / "semi-frozen4",
    "semi_frozen6": MODEL_ROOT_DIR / "semi-frozen6",
}

TEXT_COL = "texto_diario"
LABEL_COL = "emocion_principal"

LABEL_ORDER = ["joy", "sadness", "fear", "anger", "love", "surprise"]
BATCH_SIZE = 32


def _normalize_label(x: str) -> str:
    return str(x).strip().lower()


def _map_pipeline_label(label: str, id2label: dict) -> str:
    s = str(label).strip()
    if s.upper().startswith("LABEL_"):
        try:
            idx = int(s.split("_", 1)[1])
            return _normalize_label(id2label.get(idx, s))
        except Exception:
            return _normalize_label(s)
    return _normalize_label(s)


def _predict_labels(texts, model_dir: Path):
    device = 0 if torch.cuda.is_available() else -1

    config = AutoConfig.from_pretrained(model_dir)
    id2label = getattr(config, "id2label", {}) or {}

    clf = pipeline(
        "text-classification",
        model=str(model_dir),
        tokenizer=str(TOKENIZER_DIR),  # ✅ tokenizer común desde base/
        device=device,
        truncation=True,
    )

    y_pred = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        outputs = clf(batch, batch_size=BATCH_SIZE)
        for out in outputs:
            y_pred.append(_map_pipeline_label(out["label"], id2label))

    return y_pred


def _compute_metrics_and_per_class(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)

    # Global weighted
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0
    )

    metrics = {
        "accuracy": float(acc),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f1_w),
    }

    per_class = {
        label: {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1": float(f1_cls[idx]),
        }
        for idx, label in enumerate(LABEL_ORDER)
    }

    return metrics, per_class


def _evaluate_finetuned(texts, y_true):
    model_name = "finetuned"
    model_dir = MODELS["finetuned"]

    print(f"Se está haciendo la evaluación del modelo: {model_name}...")

    y_pred = _predict_labels(texts, model_dir)
    metrics, per_class = _compute_metrics_and_per_class(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {"name": model_name, "path": str(model_dir)},
        "test_set": {"path": str(TEST_CSV), "num_samples": len(y_true)},
        "labels_order": LABEL_ORDER,
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABEL_ORDER,
            "matrix": cm.tolist(),
        },
    }

    out_json = OUTPUT_DIR / "report_finetuned.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK. Guardado: {out_json}")


def _evaluate_frozen(texts, y_true):
    model_name = "frozen"
    model_dir = MODELS["frozen"]

    print(f"Se está haciendo la evaluación del modelo: {model_name}...")

    y_pred = _predict_labels(texts, model_dir)
    metrics, per_class = _compute_metrics_and_per_class(y_true, y_pred)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {"name": model_name, "path": str(model_dir)},
        "test_set": {"path": str(TEST_CSV), "num_samples": len(y_true)},
        "labels_order": LABEL_ORDER,
        "metrics": metrics,
        "per_class": per_class,
    }

    out_json = OUTPUT_DIR / "report_frozen.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK. Guardado: {out_json}")


def _evaluate_semi_frozen_variants(texts, y_true):
    print("Se está haciendo la evaluación de los modelos: semi_frozen2/4/6...")

    variants_payload = {}

    for name, model_dir in SEMI_FROZEN_VARIANTS.items():
        y_pred = _predict_labels(texts, model_dir)
        metrics, per_class = _compute_metrics_and_per_class(y_true, y_pred)

        variants_payload[name] = {
            "model": {"name": name, "path": str(model_dir)},
            "metrics": metrics,
            "per_class": per_class,
        }

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_set": {"path": str(TEST_CSV), "num_samples": len(y_true)},
        "labels_order": LABEL_ORDER,
        "variants": variants_payload,
    }

    out_json = OUTPUT_DIR / "report_semi_frozen.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"OK. Guardado: {out_json}")


def main():
    df = pd.read_csv(TEST_CSV)
    texts = df[TEXT_COL].astype(str).tolist()
    y_true = [_normalize_label(x) for x in df[LABEL_COL].tolist()]

    _evaluate_finetuned(texts, y_true)
    _evaluate_frozen(texts, y_true)
    _evaluate_semi_frozen_variants(texts, y_true)


if __name__ == "__main__":
    main()