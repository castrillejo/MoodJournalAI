import os
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILE = os.path.join(BASE_DIR, "data", "finetuning", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model-training", "download-model", "roberta-base-english", "finetuned-emotion")

EMOTION_LABELS = ['joy', 'sadness', 'fear', 'anger', 'love', 'surprise']

# ==========================================
# FUNCIONES
# ==========================================

def evaluate_model():
    print("🔍 Iniciando evaluación del modelo fine-tuned...\n")
    
    # 1. Verificar archivos
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"❌ No se encuentra el archivo de test: {TEST_FILE}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ No se encuentra el modelo entrenado: {MODEL_PATH}")
    
    # 2. Cargar test set
    print(f"📂 Cargando test set desde: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    print(f"✅ {len(test_df)} ejemplos cargados\n")
    
    # 3. Cargar modelo fine-tuned
    print(f"🤖 Cargando modelo desde: {MODEL_PATH}")
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=0 if os.system("nvidia-smi > nul 2>&1") == 0 else -1  # GPU si está disponible
    )
    print("✅ Modelo cargado\n")
    
    # 4. Hacer predicciones
    print("🔮 Realizando predicciones en test set...")
    predictions = []
    prediction_scores = []
    
    for text in test_df['texto_diario']:
        result = classifier(text, top_k=1)[0]
        predictions.append(result['label'])
        prediction_scores.append(result['score'])
    
    test_df['predicted'] = predictions
    test_df['confidence'] = prediction_scores
    
    print("✅ Predicciones completadas\n")
    
    # 5. Calcular métricas
    print("="*70)
    print("📊 RESULTADOS DE EVALUACIÓN")
    print("="*70)
    
    y_true = test_df['emocion_principal']
    y_pred = test_df['predicted']
    
    # Accuracy y F1 general
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n✨ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✨ F1-Score:  {f1:.4f}\n")
    
    # Reporte detallado por clase
    print("📋 Reporte por emoción:")
    print("-" * 70)
    print(classification_report(y_true, y_pred, labels=EMOTION_LABELS))
    
    # 6. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=EMOTION_LABELS)
    
    print("\n📈 Matriz de Confusión:")
    print("-" * 70)
    cm_df = pd.DataFrame(cm, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    print(cm_df)
    
    # 7. Guardar resultados
    results_path = os.path.join(BASE_DIR, "notebooks", "evaluation_results.csv")
    test_df.to_csv(results_path, index=False)
    print(f"\n💾 Resultados detallados guardados en: {results_path}")
    
    # 8. Visualizar matriz de confusión (opcional)
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=EMOTION_LABELS, 
                    yticklabels=EMOTION_LABELS)
        plt.title(f'Matriz de Confusión\nAccuracy: {accuracy:.2%} | F1: {f1:.4f}')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        
        plot_path = os.path.join(BASE_DIR, "notebooks", "confusion_matrix.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️  No se pudo generar el gráfico: {e}")
    
    # 9. Mostrar algunos ejemplos (aciertos y errores)
    print("\n" + "="*70)
    print("🎯 EJEMPLOS DE PREDICCIONES")
    print("="*70)
    
    # Aciertos con alta confianza
    correct = test_df[test_df['emocion_principal'] == test_df['predicted']].nlargest(3, 'confidence')
    print("\n✅ Aciertos con alta confianza:")
    for _, row in correct.iterrows():
        print(f"  - Real: {row['emocion_principal']:8} | Pred: {row['predicted']:8} ({row['confidence']:.2%})")
        print(f"    Texto: {row['texto_diario'][:100]}...")
        print()
    
    # Errores
    incorrect = test_df[test_df['emocion_principal'] != test_df['predicted']].head(3)
    if len(incorrect) > 0:
        print("❌ Ejemplos de errores:")
        for _, row in incorrect.iterrows():
            print(f"  - Real: {row['emocion_principal']:8} | Pred: {row['predicted']:8} ({row['confidence']:.2%})")
            print(f"    Texto: {row['texto_diario'][:100]}...")
            print()
    
    print("="*70)
    print(f"🎉 Evaluación completada. Meta: F1 > 0.70 | Tu resultado: {f1:.4f}")
    print("="*70)

if __name__ == "__main__":
    evaluate_model()
