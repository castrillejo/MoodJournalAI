import os
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)

# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(BASE_DIR, "data", "finetuning", "test.csv")
MODEL_ROOT = os.path.join(BASE_DIR, "model-training", "download-model", "roberta-base-english")
BASE_MODEL_PATH = os.path.join(MODEL_ROOT, "base")
FROZEN_MODEL_PATH = os.path.join(MODEL_ROOT, "frozen-classifier")
FINETUNED_MODEL_PATH = os.path.join(MODEL_ROOT, "finetuned-emotion")

EMOTION_LABELS = ['joy', 'sadness', 'fear', 'anger', 'love', 'surprise']
LABEL_MAP = {
    'joy': 0, 'sadness': 1, 'fear': 2,
    'anger': 3, 'love': 4, 'surprise': 5
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# FUNCIONES
# ==========================================

def create_untrained_classifier():
    """
    Crea un clasificador con RoBERTa-base pero con pesos ALEATORIOS
    en la capa de clasificación (simula modelo sin entrenar)
    """
    print("🔧 Creando modelo BASE con clasificador aleatorio...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ROOT)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_PATH,
        num_labels=6,
        id2label=ID2LABEL,
        label2id=LABEL_MAP
    )
    
    print("✅ Modelo base cargado (clasificador SIN entrenar)\n")
    
    return tokenizer, model

def evaluate_model(model_path, test_df, model_name, tokenizer=None, model=None):
    """Evalúa un modelo dado (puede ser una ruta o un modelo ya cargado)"""
    print(f"🔮 Evaluando modelo {model_name}...")
    
    device = 0 if torch.cuda.is_available() else -1
    
    if model_path and os.path.exists(model_path):
        classifier = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
            device=device
        )
    elif model is not None and tokenizer is not None:
        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
    else:
        print(f"⚠️  Modelo {model_name} no encontrado, saltando...")
        return None
    
    predictions = []
    for text in test_df['texto_diario']:
        result = classifier(text, top_k=1)[0]
        predictions.append(result['label'])
    
    y_true = test_df['emocion_principal']
    y_pred = predictions
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("✅ Evaluación completada\n")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': predictions,
        'report': classification_report(y_true, y_pred, labels=EMOTION_LABELS, output_dict=True)
    }

def print_comparison_3models(base_results, frozen_results, finetuned_results):
    """Imprime comparación de 3 modelos lado a lado"""
    
    print("="*120)
    print(" "*45 + "📊 COMPARACIÓN DE 3 MODELOS")
    print("="*120)
    print()
    print(f"{'Métrica':<20} {'BASE (aleatorio)':<30} {'FROZEN (solo clasif.)':<30} {'FINE-TUNED (todo)':<30}")
    print("-"*120)
    
    # Métricas generales - Accuracy
    base_acc = base_results['accuracy']
    frozen_acc = frozen_results['accuracy'] if frozen_results else 0
    ft_acc = finetuned_results['accuracy']
    
    print(f"{'Accuracy':<20} {base_acc:.4f} ({base_acc*100:.2f}%){'':<15} ", end="")
    if frozen_results:
        print(f"{frozen_acc:.4f} ({frozen_acc*100:.2f}%){'':<15} ", end="")
    print(f"{ft_acc:.4f} ({ft_acc*100:.2f}%)")
    
    # F1-Score
    base_f1 = base_results['f1']
    frozen_f1 = frozen_results['f1'] if frozen_results else 0
    ft_f1 = finetuned_results['f1']
    
    print(f"{'F1-Score (weighted)':<20} {base_f1:.4f}{'':<23} ", end="")
    if frozen_results:
        print(f"{frozen_f1:.4f}{'':<23} ", end="")
    print(f"{ft_f1:.4f}")
    print()
    
    # Por emoción
    print("="*120)
    print(" "*50 + "F1-Score por Emoción")
    print("="*120)
    print(f"{'Emoción':<20} {'BASE':<30} {'FROZEN':<30} {'FINE-TUNED':<30}")
    print("-"*120)
    
    for emotion in EMOTION_LABELS:
        base_f1_em = base_results['report'][emotion]['f1-score']
        frozen_f1_em = frozen_results['report'][emotion]['f1-score'] if frozen_results else 0
        ft_f1_em = finetuned_results['report'][emotion]['f1-score']
        
        print(f"{emotion:<20} {base_f1_em:.4f}{'':<23} ", end="")
        if frozen_results:
            print(f"{frozen_f1_em:.4f}{'':<23} ", end="")
        print(f"{ft_f1_em:.4f}")
    
    print()
    print("="*120)
    print("🎯 CONCLUSIONES:")
    print("="*120)
    print(f"1. BASE (clasificador aleatorio):          F1 = {base_f1:.4f}")
    if frozen_results:
        print(f"2. FROZEN (solo clasificador entrenado):   F1 = {frozen_f1:.4f}")
        print(f"   → Mejora vs BASE: {(frozen_f1 - base_f1):.4f} ({((frozen_f1-base_f1)/base_f1*100):.1f}%)")
    print(f"3. FINE-TUNED (todo entrenado):            F1 = {ft_f1:.4f}")
    if frozen_results:
        print(f"   → Mejora vs FROZEN: {(ft_f1 - frozen_f1):.4f} ({((ft_f1-frozen_f1)/frozen_f1*100):.1f}%)")
    print()
    print("💡 Interpretación:")
    print("   - BASE: Clasificador aleatorio, solo sirve como baseline")
    if frozen_results:
        print("   - FROZEN: Aprovecha embeddings pre-entrenados de RoBERTa (feature extraction)")
        print("   - FINE-TUNED: Además ajusta los embeddings al dominio específico (diarios)")
        print()
        if ft_f1 - frozen_f1 > 0.05:
            print("   ✅ El fine-tuning completo fue CLAVE: ajustar embeddings dio +{:.1f}% F1".format((ft_f1-frozen_f1)*100))
        else:
            print("   ✅ Los embeddings de RoBERTa ya eran buenos: frozen ≈ fine-tuned")
    else:
        print("   - FINE-TUNED: Ajusta todo el modelo al dominio específico")
    print("="*120)

def main():
    print("🚀 COMPARACIÓN: Base vs Frozen vs Fine-Tuned\n")
    
    # 1. Cargar test set
    print(f"📂 Cargando test set desde: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"✅ {len(test_df)} ejemplos cargados\n")
    
    # 2. Evaluar modelo base (sin entrenar)
    print("-"*120)
    print("FASE 1: Evaluando MODELO BASE (clasificador aleatorio)")
    print("-"*120)
    tokenizer, base_model = create_untrained_classifier()
    base_results = evaluate_model(None, test_df, "BASE", tokenizer, base_model)
    
    # 3. Evaluar modelo frozen (si existe)
    frozen_results = None
    if os.path.exists(FROZEN_MODEL_PATH):
        print("-"*120)
        print("FASE 2: Evaluando MODELO FROZEN (solo clasificador entrenado)")
        print("-"*120)
        frozen_results = evaluate_model(FROZEN_MODEL_PATH, test_df, "FROZEN")
    else:
        print("-"*120)
        print("⚠️  MODELO FROZEN no encontrado (ejecuta train_frozen.py primero)")
        print("-"*120)
        print()
    
    # 4. Evaluar modelo fine-tuned
    print("-"*120)
    print(f"FASE {3 if frozen_results else 2}: Evaluando MODELO FINE-TUNED")
    print("-"*120)
    finetuned_results = evaluate_model(FINETUNED_MODEL_PATH, test_df, "FINE-TUNED")
    
    # 5. Comparar
    if frozen_results:
        print_comparison_3models(base_results, frozen_results, finetuned_results)
    else:
        # Si no hay frozen, usa la comparación de 2 modelos
        print_comparison_2models(base_results, finetuned_results)
    
    # 6. Guardar resultados
    comparison_data = {
        'texto': test_df['texto_diario'],
        'real': test_df['emocion_principal'],
        'pred_base': base_results['predictions'],
        'pred_finetuned': finetuned_results['predictions']
    }
    
    if frozen_results:
        comparison_data['pred_frozen'] = frozen_results['predictions']
    
    comparison_df = pd.DataFrame(comparison_data)
    
    output_path = os.path.join(BASE_DIR, "notebooks", "comparison_results.csv")
    comparison_df.to_csv(output_path, index=False)
    print(f"\n💾 Comparación detallada guardada en: {output_path}")

def print_comparison_2models(base_results, finetuned_results):
    """Comparación de solo 2 modelos (fallback si no hay frozen)"""
    print("="*90)
    print(" "*25 + "📊 COMPARACIÓN DE MODELOS")
    print("="*90)
    print()
    print(f"{'Métrica':<20} {'BASE (sin entrenar)':<30} {'FINE-TUNED':<30}")
    print("-"*90)
    
    base_acc = base_results['accuracy']
    ft_acc = finetuned_results['accuracy']
    
    print(f"{'Accuracy':<20} {base_acc:.4f} ({base_acc*100:.2f}%){'':<15} {ft_acc:.4f} ({ft_acc*100:.2f}%)")
    
    base_f1 = base_results['f1']
    ft_f1 = finetuned_results['f1']
    
    print(f"{'F1-Score (weighted)':<20} {base_f1:.4f}{'':<23} {ft_f1:.4f}")
    print()
    print("="*90)
    print(f"Mejora: {(ft_f1-base_f1):.4f} ({((ft_f1-base_f1)/base_f1*100):.1f}%)")
    print("="*90)

if __name__ == "__main__":
    main()
