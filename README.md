# MoodJournalAI

## Introducción

MoodJournalAI es un sistema de análisis de emociones y estados de ánimo basado en entradas de diario personal. El proyecto utiliza procesamiento de lenguaje natural para analizar sentimientos en textos de diarios, identificando patrones emocionales y tendencias en el bienestar de los usuarios. Actualmente cuenta con una base de datos PostgreSQL y un pipeline ETL para la carga de datos de muestra.

---

## Instalación de Docker

1. Descarga Docker Desktop desde [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Ejecuta el instalador y sigue las instrucciones
3. Reinicia tu computadora si es necesario
4. Verifica la instalación:
   ```powershell
   docker --version
   ```

---

## Levantar el Proyecto

Una vez instalado Docker, ejecuta los siguientes comandos desde la raíz del proyecto:

```bash
# Construir y levantar los contenedores
docker-compose up --build
```

Para ejecutar en segundo plano:
```bash
docker-compose up --build -d
```

**Servicios disponibles:**
- PostgreSQL: `localhost:5432` (usuario: `admin`, contraseña: `admin`, base de datos: `moodjournal`)

**Comandos útiles:**
```bash
# Ver estado de los contenedores
docker-compose ps

# Ver logs
docker-compose logs -f

# Detener los contenedores
docker-compose down

# Conectarse a PostgreSQL
docker exec -it moodjournal_postgres psql -U admin -d moodjournal
```