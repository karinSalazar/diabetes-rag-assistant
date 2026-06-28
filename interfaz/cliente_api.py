"""
Cliente de la API para la interfaz Streamlit.

Encapsula todas las llamadas HTTP a la API REST del sistema.
La interfaz usa este cliente en lugar de llamar a la API directamente.

Uso:
    from interfaz.cliente_api import ClienteAPI
    cliente = ClienteAPI("http://localhost:8000")
    info = cliente.verificar_codigo("drab_...")
"""

import requests

from config import Config


class ClienteAPI:
    """Cliente para comunicarse con la API del sistema RAG."""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or Config.API_BASE).rstrip("/")

    # -- Metodos publicos (sin codigo) --

    def info_sistema(self) -> dict:
        """Obtiene la informacion general del sistema."""
        try:
            r = requests.get(f"{self.base_url}/", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def comprobar_conexion(self) -> bool:
        """Comprueba si la API esta disponible."""
        try:
            r = requests.get(f"{self.base_url}/salud", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # -- Metodos con codigo de acceso --

    def verificar_codigo(self, codigo: str) -> dict:
        """
        Verifica un codigo de acceso y devuelve el rol.
        Returns: {"valido": True/False, "nombre": ..., "rol": ...}
        """
        try:
            r = requests.get(
                f"{self.base_url}/verificar-codigo",
                headers={"X-API-Key": codigo},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                # La API devuelve {"valido": True, "nombre":..., "rol":...}
                return data
            return {"valido": False, "error": "Codigo invalido o revocado"}
        except Exception as e:
            return {"valido": False, "error": str(e)}

    def preguntar(self, pregunta: str, codigo: str, historial: list = None) -> dict:
        """
        Envia una pregunta al chatbot a traves de la API.

        Args:
            pregunta:  texto de la consulta
            codigo:    codigo de acceso del usuario
            historial: lista de mensajes previos [{"rol":..., "texto":...}]

        Returns:
            dict con respuesta, fuentes, tiempo, etc. (o {"error": ...})
        """
        try:
            r = requests.post(
                f"{self.base_url}/preguntar",
                headers={"X-API-Key": codigo},
                json={"pregunta": pregunta, "historial": historial or []},
                timeout=120,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Error {r.status_code}: {r.text}"}
        except Exception as e:
            return {"error": str(e)}

    # -- Metodos de medico --

    def listar_documentos(self, codigo: str) -> dict:
        """Lista los documentos del corpus (requiere rol medico+)."""
        try:
            r = requests.get(
                f"{self.base_url}/documentos",
                headers={"X-API-Key": codigo},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Sin permiso (codigo {r.status_code})"}
        except Exception as e:
            return {"error": str(e)}

    # -- Metodos de admin --

    def listar_codigos(self, codigo: str) -> dict:
        """Lista todos los codigos de acceso (requiere admin)."""
        try:
            r = requests.get(
                f"{self.base_url}/admin/codigos",
                headers={"X-API-Key": codigo},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Sin permiso (codigo {r.status_code})"}
        except Exception as e:
            return {"error": str(e)}

    def crear_codigo(self, codigo_admin: str, nombre: str, rol: str) -> dict:
        """Crea un nuevo codigo de acceso (requiere admin)."""
        try:
            r = requests.post(
                f"{self.base_url}/admin/codigos",
                headers={"X-API-Key": codigo_admin},
                json={"nombre": nombre, "rol": rol},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Error {r.status_code}: {r.text}"}
        except Exception as e:
            return {"error": str(e)}

    def revocar_codigo(self, codigo_admin: str, nombre: str) -> dict:
        """Revoca un codigo de acceso (requiere admin)."""
        try:
            r = requests.delete(
                f"{self.base_url}/admin/codigos",
                headers={"X-API-Key": codigo_admin},
                json={"nombre": nombre},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Error {r.status_code}: {r.text}"}
        except Exception as e:
            return {"error": str(e)}

    def estadisticas(self, codigo_admin: str) -> dict:
        """Obtiene estadisticas de uso (requiere admin)."""
        try:
            r = requests.get(
                f"{self.base_url}/admin/estadisticas",
                headers={"X-API-Key": codigo_admin},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json()
            return {"error": f"Sin permiso (codigo {r.status_code})"}
        except Exception as e:
            return {"error": str(e)}
