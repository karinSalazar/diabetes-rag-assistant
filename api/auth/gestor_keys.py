"""
Gestor de códigos de acceso (API keys) del sistema RAG.

Implementa el control de acceso basado en roles (RBAC) con tres niveles:
  - lectura  (paciente): solo puede preguntar al chatbot
  - medico:  puede preguntar y consultar los documentos del corpus
  - admin:   acceso total (gestión de accesos, subir documentos, stats)

SEGURIDAD: las claves NO se almacenan en texto plano. Se guarda su
hash SHA-256. Así, aunque alguien acceda al archivo de claves, no puede
recuperar las claves originales. Es el principio de no almacenar
secretos en claro.

Cada clave tiene el formato: drab_ + 32 bytes en hexadecimal
(drab = Diabetes RAG Assistant, prefijo identificable).

Uso:
    from api.auth.gestor_keys import crear_api_key, validar_api_key

    key = crear_api_key("Dr. García", "medico")  # se muestra una vez
    info = validar_api_key(key)  # devuelve {nombre, rol, ...} o None
"""

import hashlib
import secrets
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


# Archivo donde se guardan las claves (su hash). NUNCA al repositorio.
KEYS_FILE = Path("api/auth/keys.json")

# Roles válidos del sistema
ROLES_VALIDOS = {"lectura", "medico", "admin"}


# ── Utilidades internas ──────────────────────────────────────

def _cargar_keys() -> dict:
    """Carga el archivo de claves. Devuelve dict vacío si no existe."""
    if not KEYS_FILE.exists():
        return {}
    try:
        return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _guardar_keys(keys: dict):
    """Guarda el diccionario de claves en el archivo."""
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(
        json.dumps(keys, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _hash_key(key: str) -> str:
    """
    Calcula el hash SHA-256 de una clave.
    Esto es lo que se almacena, NUNCA la clave original.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ── Crear una clave ──────────────────────────────────────────

def crear_api_key(nombre: str, rol: str = "lectura") -> str:
    """
    Crea una nueva clave de acceso para un usuario.

    Args:
        nombre: nombre identificativo del usuario (ej. "Dr. García")
        rol:    uno de "lectura", "medico", "admin"

    Returns:
        La clave generada (en texto plano). IMPORTANTE: se muestra
        solo esta vez; después solo se guarda su hash.

    Lanza:
        ValueError si el rol no es válido.
    """
    if rol not in ROLES_VALIDOS:
        raise ValueError(
            f"Rol '{rol}' no válido. Usa uno de: {', '.join(sorted(ROLES_VALIDOS))}"
        )

    # Generar una clave aleatoria criptográficamente segura
    key = f"drab_{secrets.token_hex(32)}"

    # Guardar su HASH (no la clave original)
    keys = _cargar_keys()
    keys[_hash_key(key)] = {
        "nombre":     nombre,
        "rol":        rol,
        "activa":     True,
        "creada":     datetime.now().isoformat(timespec="seconds"),
        "ultimo_uso": None,
        "total_usos": 0,
    }
    _guardar_keys(keys)

    return key


# ── Validar una clave ────────────────────────────────────────

def validar_api_key(key: str) -> Optional[dict]:
    """
    Valida una clave de acceso.

    Args:
        key: la clave a validar

    Returns:
        dict con {nombre, rol, ...} si la clave es válida y está activa.
        None si la clave no existe o está revocada.

    Efecto secundario: actualiza la fecha de último uso y el contador.
    """
    if not key:
        return None

    hash_recibido = _hash_key(key)
    keys = _cargar_keys()

    info = keys.get(hash_recibido)
    if info is None or not info.get("activa", False):
        return None

    # Actualizar estadísticas de uso
    info["ultimo_uso"] = datetime.now().isoformat(timespec="seconds")
    info["total_usos"] = info.get("total_usos", 0) + 1
    keys[hash_recibido] = info
    _guardar_keys(keys)

    return info


# ── Revocar una clave ────────────────────────────────────────

def revocar_api_key(nombre: str) -> bool:
    """
    Revoca (desactiva) las claves de un usuario por su nombre.
    NO borra la clave: la marca como inactiva (para auditoría, RGPD art. 30).

    Returns:
        True si se revocó alguna clave, False si no se encontró.
    """
    keys = _cargar_keys()
    encontrada = False

    for hash_key, info in keys.items():
        if info.get("nombre") == nombre and info.get("activa"):
            info["activa"] = False
            info["revocada"] = datetime.now().isoformat(timespec="seconds")
            encontrada = True

    if encontrada:
        _guardar_keys(keys)

    return encontrada


# ── Listar claves (para el admin) ────────────────────────────

def listar_keys() -> list[dict]:
    """
    Devuelve la lista de todas las claves (sin exponer el hash completo).
    Para el panel de administración.
    """
    keys = _cargar_keys()
    resultado = []
    for hash_key, info in keys.items():
        resultado.append({
            "nombre":     info.get("nombre"),
            "rol":        info.get("rol"),
            "activa":     info.get("activa"),
            "creada":     info.get("creada"),
            "ultimo_uso": info.get("ultimo_uso"),
            "total_usos": info.get("total_usos", 0),
            # Solo mostramos los primeros caracteres del hash (identificación)
            "id_hash":    hash_key[:12] + "...",
        })
    return resultado


# ── Prueba directa del módulo ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DEL GESTOR DE CÓDIGOS DE ACCESO")
    print("=" * 60)

    # Crear una clave de cada rol
    print("\nCreando claves de prueba...")
    key_paciente = crear_api_key("Paciente Demo", "lectura")
    key_medico   = crear_api_key("Dr. Demo", "medico")
    key_admin    = crear_api_key("Admin Demo", "admin")

    print(f"  Paciente: {key_paciente}")
    print(f"  Médico:   {key_medico}")
    print(f"  Admin:    {key_admin}")

    # Validar una clave
    print("\nValidando la clave del médico...")
    info = validar_api_key(key_medico)
    print(f"  Resultado: {info['nombre']} (rol: {info['rol']})")

    # Validar una clave inválida
    print("\nValidando una clave inventada...")
    info_falsa = validar_api_key("drab_clave_falsa_123")
    print(f"  Resultado: {info_falsa}")  # debe ser None

    # Listar claves
    print("\nClaves registradas:")
    for k in listar_keys():
        estado = "activa" if k["activa"] else "revocada"
        print(f"  - {k['nombre']} ({k['rol']}) [{estado}] usos: {k['total_usos']}")

    print("\n(Nota: esto creó un archivo api/auth/keys.json de prueba.")
    print(" Bórralo antes de crear las claves reales con crear_admin.py)")