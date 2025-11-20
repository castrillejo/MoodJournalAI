FROM python:3.10-slim

WORKDIR /app

# Copiar requirements desde la ruta correcta (contexto = raíz del proyecto)
COPY etl/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar script desde la ruta correcta
COPY etl/load_data.py .

# Copiar la carpeta data completa
COPY data /data

CMD ["python", "load_data.py"]

