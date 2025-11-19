from faker import Faker
import random
from datetime import datetime, timedelta
from datasets import load_dataset
import pandas as pd
import uuid

# ----- CONFIGURACIÓN -----
NUM_USUARIOS = 100
FECHA_INICIO = datetime(2025, 7, 1)
FECHA_FIN = datetime(2025, 9, 30)
OUTPUT_USUARIOS = "usuarios.csv"
OUTPUT_ENTRADAS = "entradas.csv"

fake = Faker("es_ES")

# ----- CARGAR DATASET -----
print("📥 Cargando dataset dair-ai/emotion...")
dataset = load_dataset("dair-ai/emotion", split="train")

# Creamos una lista [(texto, emoción)]
textos_emocion = list(zip(dataset["text"], dataset["label"]))
label_names = dataset.features["label"].names  # ['sadness','joy','love','anger','fear','surprise']

print(f"Dataset cargado con {len(textos_emocion)} ejemplos y etiquetas {label_names}")

# ----- GENERACIÓN DE USUARIOS -----
OCUPACIONES = ["Estudiante", "Empleado", "Freelance", "Desempleado", "Jubilado"]
PERSONALIDADES = ["Optimista", "Melancólico", "Ansioso", "Empático", "Introvertido", "Extrovertido"]

def generar_usuario():
    return {
        "id_usuario": str(uuid.uuid4()),
        "nombre": fake.first_name(),
        "sexo": random.choice(["M", "F"]),
        "edad": random.randint(18, 70),
        "ocupacion": random.choice(OCUPACIONES),
        "personalidad": random.choice(PERSONALIDADES),
        "p_actividad": round(random.uniform(0.4, 0.9), 2),
    }

usuarios = [generar_usuario() for _ in range(NUM_USUARIOS)]

# ----- GENERAR ENTRADAS -----
def generar_fechas(inicio, fin, p_activo):
    fechas = []
    actual = inicio
    while actual <= fin:
        if random.random() < p_activo:
            fechas.append(actual)
        actual += timedelta(days=1)
    return fechas

entradas = []

for user in usuarios:
    fechas = generar_fechas(FECHA_INICIO, FECHA_FIN, user["p_actividad"])
    for fecha in fechas:
        texto, label = random.choice(textos_emocion)
        emocion = label_names[label]
        entradas.append({
            "id_entrada": str(uuid.uuid4()),
            "id_usuario": user["id_usuario"],
            "fecha": fecha.strftime("%Y-%m-%d"),
            "texto_diario": texto,
            "emocion_principal": emocion,
            "sentimiento_usuario": "",
        })

# ----- EXPORTAR -----
df_usuarios = pd.DataFrame(usuarios)
df_entradas = pd.DataFrame(entradas)

df_usuarios.to_csv(OUTPUT_USUARIOS, index=False, encoding="utf-8")
df_entradas.to_csv(OUTPUT_ENTRADAS, index=False, encoding="utf-8")

print(f"✅ {len(usuarios)} usuarios guardados en {OUTPUT_USUARIOS}")
print(f"✅ {len(entradas)} entradas guardadas en {OUTPUT_ENTRADAS}")
print(f"📊 Emociones generadas: {df_entradas['emocion_principal'].value_counts().to_dict()}")
