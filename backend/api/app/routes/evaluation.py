import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter()

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets" / "evaluation"

REPORT_PATHS = {
    "frozen": ASSETS_DIR / "report_frozen.json",
    "semi_frozen": ASSETS_DIR / "report_semi_frozen.json",
    "finetuned": ASSETS_DIR / "report_finetuned.json",
}

def _load_json(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No existe el archivo: {path}")

def _minimal_report(model_name: str, report: dict, labels_order: list) -> dict:
    return {
        "model": {"name": model_name},
        "metrics": report.get("metrics", {}),
        "per_class": report.get("per_class", {}),
        "labels_order": labels_order,
    }

@router.get("/evaluation/overview")
def evaluation_overview():
    frozen = _load_json(REPORT_PATHS["frozen"])
    semi = _load_json(REPORT_PATHS["semi_frozen"])
    fin = _load_json(REPORT_PATHS["finetuned"])

    # orden de etiquetas: preferimos finetuned, si no semi, si no frozen
    labels_order = (
        fin.get("labels_order")
        or semi.get("labels_order")
        or frozen.get("labels_order")
        or []
    )

    # Confusion matrix SOLO del finetuned (valores para renderizar en frontend)
    fin_conf = fin.get("confusion_matrix", {}) or {}

    # Variantes del semi-frozen vienen en tu JSON como:
    # { "variants": { "semi_frozen2": {...}, "semi_frozen4": {...}, "semi_frozen6": {...} } }
    variants = semi.get("variants", {}) or {}

    return {
        "models": {
            "frozen": _minimal_report("frozen", frozen, labels_order),
            "finetuned": _minimal_report("finetuned", fin, labels_order),
            "semi_frozen": {
                "model": {"name": "semi_frozen"},
                "labels_order": labels_order,
                "variants": {
                    k: _minimal_report(k, v, labels_order)
                    for k, v in variants.items()
                },
            },
        },
        "confusion_matrix": {
            "labels": fin_conf.get("labels", labels_order),
            "matrix": fin_conf.get("matrix", []),
        },
    }
