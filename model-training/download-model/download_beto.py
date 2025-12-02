"""
Script para descargar el modelo BETO (BERT Español) pre-entrenado
y guardarlo localmente en esta carpeta.

Modelo: dccuchile/bert-base-spanish-wwm-cased
Uso: python download_beto.py
"""

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import os
from pathlib import Path

# Configuración
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
SAVE_DIR = Path(__file__).parent / "bert-base-spanish"

def download_model():
    """
    Descarga el modelo BETO y el tokenizer desde Hugging Face Hub
    y los guarda en la carpeta local.
    """
    print("=" * 60)
    print("📥 DESCARGANDO MODELO BETO ESPAÑOL")
    print("=" * 60)
    print(f"\n🔍 Modelo: {MODEL_NAME}")
    print(f"💾 Guardando en: {SAVE_DIR.absolute()}\n")
    
    # Crear directorio si no existe
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Descargar Tokenizer
        print("📚 Descargando tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(SAVE_DIR)
        print("✅ Tokenizer descargado y guardado\n")
        
        # 2. Descargar Modelo Base (para fine-tuning desde cero)
        print("🤖 Descargando modelo base...")
        model_base = AutoModel.from_pretrained(MODEL_NAME)
        model_base.save_pretrained(SAVE_DIR / "base")
        print("✅ Modelo base descargado y guardado\n")
        
        # 3. Información del modelo
        print("=" * 60)
        print("✨ DESCARGA COMPLETADA")
        print("=" * 60)
        print(f"\n📦 Archivos guardados en: {SAVE_DIR.absolute()}\n")
        
        # Listar archivos
        print("📂 Estructura de archivos:")
        print(f"\n{SAVE_DIR}/")
        for item in SAVE_DIR.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                relative_path = item.relative_to(SAVE_DIR)
                print(f"  ├── {relative_path} ({size_mb:.2f} MB)")
        
        # Información de uso
        print("\n" + "=" * 60)
        print("🚀 CÓMO USAR EL MODELO")
        print("=" * 60)
        print("""
from transformers import AutoTokenizer, AutoModel

# Cargar desde la carpeta local
model_path = "./bert-base-spanish/base"
tokenizer_path = "./bert-base-spanish"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(model_path)

# Ya puedes usar el modelo
text = "Hoy me siento muy feliz"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
        """)
        
        print("\n✅ ¡Todo listo para empezar el fine-tuning!\n")
        
    except Exception as e:
        print(f"\n❌ Error durante la descarga: {str(e)}")
        print("\n💡 Soluciones posibles:")
        print("  1. Verifica tu conexión a internet")
        print("  2. Instala las dependencias: pip install transformers torch")
        print("  3. Verifica que tienes espacio en disco (~500 MB)\n")
        return False
    
    return True

def verify_model():
    """
    Verifica que el modelo se descargó correctamente y puede cargarse.
    """
    print("\n🔍 Verificando modelo descargado...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
        model = AutoModel.from_pretrained(SAVE_DIR / "base")
        
        # Test rápido
        test_text = "Este es un texto de prueba"
        inputs = tokenizer(test_text, return_tensors="pt")
        outputs = model(**inputs)
        
        print("✅ Modelo verificado correctamente")
        print(f"📊 Dimensiones del output: {outputs.last_hidden_state.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error al verificar el modelo: {str(e)}")
        return False

if __name__ == "__main__":
    # Descargar modelo
    success = download_model()
    
    # Verificar descarga
    if success:
        verify_model()
