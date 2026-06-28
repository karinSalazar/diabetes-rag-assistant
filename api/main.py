"""
API REST del sistema RAG sanitario (FastAPI).

Expone el motor RAG como un servicio web con autenticación por roles.
Conecta todas las piezas construidas: anonimización, guardrails, RAG,
re-ranking, generación con Llama y trazabilidad MLflow.

Endpoints por nivel de acceso:
  PÚBLICOS:
    GET  /             → info del sistema
    GET  /salud        → comprobación de estado
  LECTURA (paciente+):
    GET  /verificar-codigo → valida el propio código y devuelve el rol
    POST /preguntar        → consulta al chatbot
  MÉDICO (médico+):
    GET  /documentos       → lista de documentos del corpus
  ADMIN:
    GET    /admin/codigos      → lista de códigos de acceso
    POST   /admin/codigos      → crea un código nuevo
    DELETE /admin/codigos      → revoca un código
    GET    /admin/estadisticas → estadísticas de uso

Arrancar con:
    python -m uvicorn api.main:app --reload
    o
    python run_api.py
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import Config
from ingesta.generador import responder
from ingesta.indexador import _obtener_coleccion
from trazabilidad.registro import inicializar_mlflow
from api.auth.dependencias import (
    requiere_lectura, requiere_medico, requiere_admin,
)
from api.auth.gestor_keys import (
    crear_api_key, revocar_api_key, listar_keys,
)


# ── Ciclo de vida de la aplicación ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Se ejecuta al arrancar y al cerrar la API."""
    print("Arrancando API del sistema RAG...")
    inicializar_mlflow()
    # Verificar que ChromaDB tiene documentos indexados
    try:
        col = _obtener_coleccion(reiniciar=False)
        print(f"ChromaDB: {col.count()} fragmentos indexados.")
    except Exception as e:
        print(f"Aviso ChromaDB: {e}")
    print("API lista.")
    yield
    print("Cerrando API.")


# ── Instancia de la aplicación ───────────────────────────────

app = FastAPI(
    title="Sistema RAG Sanitario — Diabetes",
    description="Asistente clínico conversacional para diabetes mellitus. "
                "Con anonimización de datos, guardrails de seguridad, "
                "control de acceso por roles y trazabilidad.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: permite que la interfaz Streamlit (otro origen) llame a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Modelos de datos (validación automática) ─────────────────

class MensajeHistorial(BaseModel):
    rol:   str   # "usuario" o "asistente"
    texto: str

class PreguntaEntrada(BaseModel):
    pregunta:  str = Field(..., min_length=3, max_length=Config.MAX_PREGUNTA_LEN)
    historial: list[MensajeHistorial] = []

class RespuestaSalida(BaseModel):
    respuesta:       str
    fuentes:         list[str]
    n_fragmentos:    int
    tiempo_seg:      float
    bloqueado:       bool
    categoria:       str
    pii_enmascarada: int
    consultado_por:  str

class NuevoCodigoEntrada(BaseModel):
    nombre: str = Field(..., min_length=1)
    rol:    str = "lectura"

class RevocarCodigoEntrada(BaseModel):
    nombre: str


# ── Endpoints públicos ───────────────────────────────────────

@app.get("/")
def raiz():
    """Información general del sistema."""
    try:
        col = _obtener_coleccion(reiniciar=False)
        n_fragmentos = col.count()
    except Exception:
        n_fragmentos = 0

    return {
        "servicio":             "Sistema RAG Sanitario - Diabetes",
        "version":              "1.0.0",
        "estado":               "activo",
        "modelo_llm":           Config.LLM_MODEL,
        "modelo_embedding":     Config.EMBEDDING_MODEL,
        "fragmentos_indexados": n_fragmentos,
    }


@app.get("/salud")
def salud():
    """Comprobación rápida de que la API responde."""
    return {"estado": "ok"}


# ── Endpoints de LECTURA (paciente+) ─────────────────────────

@app.get("/verificar-codigo")
def verificar_codigo(usuario: dict = Depends(requiere_lectura)):
    """Valida el código de acceso del usuario y devuelve su rol."""
    return {
        "valido": True,
        "nombre": usuario["nombre"],
        "rol":    usuario["rol"],
    }


@app.post("/preguntar", response_model=RespuestaSalida)
def preguntar(
    datos: PreguntaEntrada,
    usuario: dict = Depends(requiere_lectura),
):
    """
    Consulta principal al chatbot.
    Pasa por: anonimización → guardrails → RAG → Llama → MLflow.
    Disponible para todos los roles (paciente, médico, admin).
    """
    t0 = time.time()

    # Convertir el historial al formato que espera el generador
    historial = [{"rol": m.rol, "texto": m.texto} for m in datos.historial]

    try:
        resultado = responder(datos.pregunta, historial=historial)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar la respuesta: {e}",
        )

    return RespuestaSalida(
        respuesta=resultado["respuesta"],
        fuentes=resultado["fuentes"],
        n_fragmentos=resultado["n_fragmentos"],
        tiempo_seg=round(time.time() - t0, 2),
        bloqueado=resultado.get("bloqueado", False),
        categoria=resultado.get("categoria_bloqueo", "ok"),
        pii_enmascarada=len(resultado.get("pii_detectada", [])),
        consultado_por=usuario["nombre"],
    )


# ── Endpoints de MÉDICO (médico+) ────────────────────────────

@app.get("/documentos")
def documentos(usuario: dict = Depends(requiere_medico)):
    """
    Lista los documentos del corpus clínico.
    Solo para médicos y administradores.
    """
    try:
        col = _obtener_coleccion(reiniciar=False)
        datos = col.get(include=["metadatas"])
        fuentes = sorted({
            m.get("fuente", "") for m in datos.get("metadatas", [])
            if m.get("fuente")
        })
        return {
            "total_fragmentos": col.count(),
            "total_documentos": len(fuentes),
            "documentos":       fuentes,
        }
    except Exception as e:
        raise HTTPException(500, f"Error al consultar documentos: {e}")


# ── Endpoints de ADMIN ───────────────────────────────────────

@app.get("/admin/codigos")
def admin_listar_codigos(usuario: dict = Depends(requiere_admin)):
    """Lista todos los códigos de acceso (sin exponer las claves)."""
    return {"codigos": listar_keys()}


@app.post("/admin/codigos")
def admin_crear_codigo(
    datos: NuevoCodigoEntrada,
    usuario: dict = Depends(requiere_admin),
):
    """Crea un nuevo código de acceso. Devuelve la clave (solo esta vez)."""
    try:
        clave = crear_api_key(datos.nombre, datos.rol)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "mensaje": "Código creado. Guárdalo ahora, no se mostrará de nuevo.",
        "codigo":  clave,
        "nombre":  datos.nombre,
        "rol":     datos.rol,
    }


@app.delete("/admin/codigos")
def admin_revocar_codigo(
    datos: RevocarCodigoEntrada,
    usuario: dict = Depends(requiere_admin),
):
    """Revoca (desactiva) el código de un usuario."""
    if not revocar_api_key(datos.nombre):
        raise HTTPException(404, f"No se encontró un código activo para '{datos.nombre}'.")
    return {"mensaje": f"Código de '{datos.nombre}' revocado."}


@app.get("/admin/estadisticas")
def admin_estadisticas(usuario: dict = Depends(requiere_admin)):
    """Estadísticas de uso de los códigos de acceso."""
    codigos = listar_keys()
    return {
        "total_codigos":   len(codigos),
        "codigos_activos": sum(1 for c in codigos if c["activa"]),
        "total_consultas": sum(c["total_usos"] for c in codigos),
        "detalle":         codigos,
    }