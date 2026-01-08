FROM python:3.10-slim

WORKDIR /app

COPY etl/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY etl/load_data.py .
COPY data /data

CMD ["python", "load_data.py"]