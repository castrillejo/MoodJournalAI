import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_data():
    # 1. Configuración de rutas (relativas a notebooks/)
    # Base dir es un nivel arriba de notebooks (c:\MoodJournalAI)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "entradas.csv")
    output_dir = os.path.join(base_dir, "data", "finetuning")

    print(f"📖 Leyendo archivo desde: {input_file}")
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Carpeta creada: {output_dir}")

    # 2. Cargar datos
    df = pd.read_csv(input_file)
    print(f"✅ Total entradas cargadas: {len(df)}")

    # 3. Limpieza y preparación
    # Nos interesan el texto y la emoción
    df = df[['texto_diario', 'emocion_principal']].copy()
    
    # Eliminar vacíos
    df = df.dropna()
    df = df[df['texto_diario'].str.strip() != ""]
    
    print(f"✅ Entradas válidas para entrenamiento: {len(df)}")

    # Mapeo de emociones a números (Label Encoding)
    emotion_map = {
        'joy': 0,
        'sadness': 1,
        'fear': 2,
        'anger': 3,
        'love': 4,
        'surprise': 5
    }
    
    # Filtrar solo las emociones que nos interesan (por seguridad)
    df = df[df['emocion_principal'].isin(emotion_map.keys())]
    
    # Crear columna 'label' numérica
    df['label'] = df['emocion_principal'].map(emotion_map)

    # 4. División de datos (Train 80% / Val 10% / Test 10%)
    # Primero separamos Test (10%) del resto (90%)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.1, 
        stratify=df['label'], # Mantiene proporción de emociones
        random_state=42
    )
    
    # Del resto (90%), separamos Validation (10% del total original aprox, o sea 1/9 del resto)
    # 0.1 / 0.9 = 0.111...
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=0.1111,
        stratify=train_val_df['label'],
        random_state=42
    )

    print("\n📊 Distribución final:")
    print(f"  - Train sets: {len(train_df)} entradas (Entrenamiento)")
    print(f"  - Val set:    {len(val_df)} entradas (Validación durante training)")
    print(f"  - Test set:   {len(test_df)} entradas (Evaluación final)")

    # 5. Guardar archivos
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n💾 Archivos guardados en: {output_dir}")
    print("  -> train.csv")
    print("  -> val.csv")
    print("  -> test.csv")

if __name__ == "__main__":
    prepare_data()
