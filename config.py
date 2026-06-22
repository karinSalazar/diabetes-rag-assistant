"""
Configuración centralizada del sistema RAG sanitario.

Toda la configuración del proyecto vive aquí. Los valores sensibles
(credenciales) se leen del archivo .env y NUNCA se escriben en el código.

Uso:
    from config import Config
    print(Config.LLM_MODEL)
"""

import os
from dotenv import load_dotenv

# Carga las variables del archivo .env al entorno
load_dotenv()


class Config:
    """Configuración global del sistema. Valores leídos de .env con defaults."""

    # ── AWS S3 (almacenamiento del corpus público) ──────────
    AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION            = os.getenv("AWS_REGION", "eu-west-1")
    S3_BUCKET_NAME        = os.getenv("S3_BUCKET_NAME", "diabetes-rag-docs")
    S3_RAW_PREFIX         = "raw/"          # documentos originales
    S3_PROCESSED_PREFIX   = "processed/"    # chunks procesados

    # ── Databricks (procesado distribuido del corpus) ───────
    DATABRICKS_HOST       = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN      = os.getenv("DATABRICKS_TOKEN")
    DATABRICKS_JOB_ID     = os.getenv("DATABRICKS_JOB_ID", "")

    # ── Modelos locales (Ollama) ────────────────────────────
    # LLM que genera las respuestas y modelo que crea los embeddings.
    # Ambos corren en local — los datos sensibles nunca salen del servidor.
    LLM_MODEL             = os.getenv("LLM_MODEL", "llama3.2")
    EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_HOST           = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # ── Base de datos vectorial (ChromaDB) ──────────────────
    VECTORDB_PATH         = os.getenv("VECTORDB_PATH", "./vectordb/chroma_db")
    COLLECTION_NAME       = os.getenv("COLLECTION_NAME", "diabetes_docs")

    # ── Parámetros del RAG ──────────────────────────────────
    # CHUNK_SIZE: tamaño de cada fragmento de texto (en caracteres)
    # CHUNK_OVERLAP: solapamiento entre fragmentos (preserva contexto)
    # N_RESULTADOS_RAG: fragmentos finales que se pasan al LLM
    # N_CANDIDATOS_RERANK: fragmentos que recupera ChromaDB antes del re-ranking
    CHUNK_SIZE            = 500
    CHUNK_OVERLAP         = 50
    N_RESULTADOS_RAG      = 4
    N_CANDIDATOS_RERANK   = 8

    # Modelo de re-ranking (CrossEncoder)
    RERANK_MODEL          = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Rutas locales de datos ──────────────────────────────
    DATA_RAW_PATH         = os.getenv("DATA_RAW_PATH", "./data/raw")
    DATA_PROCESSED_PATH   = os.getenv("DATA_PROCESSED_PATH", "./data/processed")

    # ── MLflow (trazabilidad) ───────────────────────────────
    MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    MLFLOW_EXPERIMENT     = "diabetes_rag_chatbot"

    # ── API ─────────────────────────────────────────────────
    API_HOST              = os.getenv("API_HOST", "0.0.0.0")
    API_PORT              = int(os.getenv("API_PORT", "8000"))
    API_BASE              = os.getenv("API_BASE", "http://localhost:8000")

    # ── Anonimización PII ───────────────────────────────────
    # Idioma para el motor NER de Presidio/spaCy
    PII_LANGUAGE          = "es"
    PII_SPACY_MODEL       = "es_core_news_md"

    # ── Límites de seguridad ────────────────────────────────
    MAX_PREGUNTA_LEN      = 1000     # caracteres máximos por pregunta
    MAX_ARCHIVO_MB        = 100      # tamaño máximo de archivo a subir

    @classmethod
    def validar(cls) -> list[str]:
        """
        Comprueba que las variables críticas estén configuradas.
        Devuelve una lista de avisos (vacía si todo está bien).
        Útil para detectar un .env mal configurado al arrancar.
        """
        avisos = []
        if not cls.AWS_ACCESS_KEY_ID:
            avisos.append("AWS_ACCESS_KEY_ID no configurado (necesario para S3)")
        if not cls.DATABRICKS_HOST:
            avisos.append("DATABRICKS_HOST no configurado (necesario para Databricks)")
        return avisos