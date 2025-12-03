from transformers import AutoTokenizer, AutoModel
import torch

print("=" * 70)
print("🎭 PRUEBA DE ANÁLISIS DE SENTIMIENTOS CON ROBERTA-BASE (INGLÉS)")
print("=" * 70)

def test_with_roberta_base():
    """
    Demuestra que RoBERTa base solo genera embeddings, no clasificaciones.
    """
    print("\n" + "=" * 70)
    print("🔍 PROBANDO CON ROBERTA-BASE (Tu modelo descargado)")
    print("=" * 70)
    
    try:
        base_path = "../model-training/download-model/roberta-base-english"
        
        print(f"\n📂 Cargando modelo local: {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModel.from_pretrained(f"{base_path}/base")
        
        print("✅ Modelo RoBERTa-base cargado\n")
        
        # Frase de prueba en inglés
        text = "Today I feel very happy"
        print(f"📝 Frase de prueba: \"{text}\"\n")
        
        # Tokenizar y obtener embeddings
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        
        # RoBERTa base genera embeddings (no clasificaciones)
        embeddings = outputs.last_hidden_state
        
        print("🎯 RESULTADOS:")
        print(f"  ├─ Dimensiones del embedding: {embeddings.shape}")
        print(f"  ├─ Tokens procesados: {len(inputs['input_ids'][0])}")
        print(f"  └─ Vector por token: {embeddings.shape[-1]} dimensiones")
        
        print("\n" + "=" * 70)
        print("💡 IMPORTANTE:")
        print("=" * 70)
        print("✅ RoBERTa-base genera embeddings (representaciones numéricas)")
        print("❌ NO clasifica sentimientos directamente")
        print("🎯 Para clasificar 6 emociones, necesitas hacer FINE-TUNING")
        print("\nEmociones objetivo:")
        print("  1. joy (alegría)")
        print("  2. sadness (tristeza)")
        print("  3. fear (miedo)")
        print("  4. anger (ira)")
        print("  5. love (amor)")
        print("  6. surprise (sorpresa)")
        print("=" * 70)
        
    except FileNotFoundError:
        print("\n❌ ERROR: No se encontró el modelo RoBERTa")
        print("💡 Debes descargar el modelo primero:")
        print("   1. cd model-training/download-model")
        print("   2. python download_roberta.py\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")

if __name__ == "__main__":
    test_with_roberta_base()
