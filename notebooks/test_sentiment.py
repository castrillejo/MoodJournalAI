from transformers import pipeline, AutoTokenizer, AutoModel
import torch

print("=" * 70)
print("🎭 PRUEBA DE ANÁLISIS DE SENTIMIENTOS EN ESPAÑOL")
print("=" * 70)

# ============================================================================
# OPCIÓN B: Mostrar que BETO base solo genera embeddings
# ============================================================================
def test_with_beto_base():
    """
    Demuestra que BETO base solo genera embeddings, no clasificaciones.
    """
    print("\n" + "=" * 70)
    print("🔍 PROBANDO CON BETO BASE (Tu modelo descargado)")
    print("=" * 70)
    
    try:
        base_path = "../model-training/download-model/bert-base-spanish"
        
        print(f"\n📂 Cargando modelo local: {base_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModel.from_pretrained(f"{base_path}/base")
        
        print("✅ Modelo BETO base cargado\n")
        
        # Frase de prueba
        text = "Hoy me siento muy feliz"
        print(f"📝 Frase de prueba: \"{text}\"\n")
        
        # Tokenizar y procesar
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        print("🔢 Tokens generados:")
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"   {tokens}\n")
        
        # Obtener embeddings
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Formato del output
        embedding_shape = outputs.last_hidden_state.shape
        
        print("📊 Output del modelo:")
        print(f"   Dimensiones: {embedding_shape}")
        print(f"   - {embedding_shape[0]} textos")
        print(f"   - {embedding_shape[1]} tokens")
        print(f"   - {embedding_shape[2]} dimensiones por token")
        
        print("\n⚠️  IMPORTANTE:")
        print("   Este es solo un EMBEDDING (representación numérica del texto).")
        print("   NO es una clasificación de sentimiento.")
        print("\n   Para clasificar sentimientos, necesitas:")
        print("   1. Fine-tunear BETO con datos etiquetados de sentimientos")
        print("   2. O usar un modelo ya fine-tuned (como en la Opción A)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    test_with_beto_base()
