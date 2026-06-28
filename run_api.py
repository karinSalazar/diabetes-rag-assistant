"""
Arranca la API del sistema RAG.

Uso:
    python run_api.py

La API quedará disponible en http://localhost:8000
La documentación interactiva en http://localhost:8000/docs
"""

import warnings
warnings.filterwarnings("ignore")

import uvicorn

from config import Config


if __name__ == "__main__":
    print("=" * 60)
    print("Iniciando Sistema RAG Sanitario - Diabetes")
    print("=" * 60)
    print(f"API:           http://localhost:{Config.API_PORT}")
    print(f"Documentación: http://localhost:{Config.API_PORT}/docs")
    print("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
    )