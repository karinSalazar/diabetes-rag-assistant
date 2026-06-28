"""
Dependencias de seguridad de la API (FastAPI).

Son los "porteros" que se ejecutan antes de cada endpoint para:
  1. Comprobar que el código de acceso (API key) es válido
  2. Comprobar que el rol del usuario tiene permiso para ese endpoint

Se usan con Depends() en los endpoints de FastAPI. Si la validación
falla, FastAPI devuelve automáticamente un error 403 (prohibido) y
el endpoint nunca llega a ejecutarse.

Jerarquía de permisos (cada rol incluye los de abajo):
    admin   → puede todo
    medico  → puede lo de médico y lo de lectura
    lectura → solo lo de lectura (paciente)

El código de acceso se envía en la cabecera HTTP "X-API-Key".

Uso en un endpoint:
    @app.post("/preguntar")
    def preguntar(datos: Pregunta, usuario: dict = Depends(requiere_lectura)):
        # Si llegamos aquí, el usuario está autenticado y autorizado
        ...
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from api.auth.gestor_keys import validar_api_key


# Define que el código de acceso viaja en la cabecera "X-API-Key"
# auto_error=False: gestionamos nosotros el error con mensajes claros
esquema_api_key = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Validación base ──────────────────────────────────────────

def _validar_y_obtener_usuario(api_key: str) -> dict:
    """
    Valida el código de acceso y devuelve la info del usuario.
    Lanza 403 si el código falta o es inválido.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Falta el código de acceso. Envíalo en la cabecera 'X-API-Key'.",
        )

    info = validar_api_key(api_key)
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Código de acceso inválido o revocado.",
        )

    return info


# ── Portero de LECTURA (paciente) ────────────────────────────

def requiere_lectura(api_key: str = Security(esquema_api_key)) -> dict:
    """
    Permite el acceso a cualquier usuario con código válido.
    Es el nivel mínimo: pacientes, médicos y admins pasan.
    """
    return _validar_y_obtener_usuario(api_key)


# ── Portero de MÉDICO ────────────────────────────────────────

def requiere_medico(api_key: str = Security(esquema_api_key)) -> dict:
    """
    Permite el acceso solo a médicos y admins.
    Bloquea a los pacientes (rol lectura).
    """
    info = _validar_y_obtener_usuario(api_key)
    if info["rol"] not in {"medico", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Acceso restringido. Tu rol '{info['rol']}' no tiene permiso "
                   "para esta acción (requiere médico o administrador).",
        )
    return info


# ── Portero de ADMIN ─────────────────────────────────────────

def requiere_admin(api_key: str = Security(esquema_api_key)) -> dict:
    """
    Permite el acceso solo a administradores.
    Bloquea a pacientes y médicos.
    """
    info = _validar_y_obtener_usuario(api_key)
    if info["rol"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso restringido a administradores.",
        )
    return info