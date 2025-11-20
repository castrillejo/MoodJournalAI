import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

# Config de conexión al postgres del docker compose
DB_CONFIG = {
    "host": "postgres",
    "database": "moodjournal",
    "user": "admin",
    "password": "admin",
    "port": 5432
}

def load_csv_to_table(csv_path, table_name, columns):
    df = pd.read_csv(csv_path)

    # Convertimos NaN a None (PostgreSQL friendly)
    df = df.where(pd.notnull(df), None)

    print(f"Cargando {len(df)} filas en {table_name}...")

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = f"""
        INSERT INTO {table_name} ({",".join(columns)})
        VALUES %s
        ON CONFLICT DO NOTHING;
    """

    values = [tuple(row[col] for col in columns) for _, row in df.iterrows()]
    execute_values(cur, insert_query, values)

    conn.commit()
    cur.close()
    conn.close()

    print(f" ✔ Carga completa: {table_name}")

def main():
    print("Iniciando ETL...")

    load_csv_to_table(
        "/data/usuarios.csv",
        "usuarios",
        ["id_usuario", "nombre", "sexo", "edad", "ocupacion", "personalidad", "p_actividad"]
    )

    load_csv_to_table(
        "/data/entradas.csv",
        "entradas_diario",
        ["id_entrada", "id_usuario", "fecha", "texto_diario", "emocion_principal", "sentimiento_usuario"]
    )

    print("ETL completada con éxito.")

if __name__ == "__main__":
    main()
