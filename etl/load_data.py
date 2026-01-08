import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os

#Datos de la BD
DB_CONFIG = {
    "host": "localhost",
    "database": "moodjournal",
    "user": "admin",
    "password": "admin",
    "port": 5432
}

def load_csv_to_table(csv_path, table_name, columns):
    
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el archivo: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    df = df.where(pd.notnull(df), None)

    try:
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

        print(f"Carga completa: {table_name}")
    except Exception as e:
        print(f"Error conectando a BD: {e}")

def main():
    print("Iniciando ETL...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    print(f"Buscando datos en: {data_dir}")

    load_csv_to_table(
        os.path.join(data_dir, "usuarios.csv"),
        "usuarios",
        ["id_usuario", "nombre", "sexo", "edad", "ocupacion", "personalidad", "p_actividad"]
    )

    load_csv_to_table(
        os.path.join(data_dir, "entradas.csv"),
        "entradas_diario",
        ["id_entrada", "id_usuario", "fecha", "texto_diario", "emocion_principal", "sentimiento_usuario"]
    )
    print("ETL completada con éxito.")

if __name__ == "__main__":
    main()