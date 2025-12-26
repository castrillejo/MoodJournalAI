from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import predict
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Crear aplicación FastAPI
app = FastAPI(
    title="MoodJournalAI API",
    description="API para clasificación de emociones usando RoBERTa fine-tuned",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS (para permitir requests desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción: especificar dominios exactos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(
    predict.router, 
    prefix="/api", 
    tags=["Predictions"]
)

@app.get("/")
async def root():
    """Endpoint raíz - información básica de la API"""
    return {
        "message": "MoodJournalAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MoodJournalAI API"
    }

@app.on_event("startup")
async def startup_event():
    """Se ejecuta al iniciar la aplicación"""
    logging.info("🚀 Iniciando MoodJournalAI API...")
    # El modelo se carga al hacer la primera predicción (lazy loading)

@app.on_event("shutdown")
async def shutdown_event():
    """Se ejecuta al cerrar la aplicación"""
    logging.info("👋 Cerrando MoodJournalAI API...")
