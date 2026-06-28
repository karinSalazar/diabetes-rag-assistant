"""
Interfaz web del sistema RAG sanitario (Streamlit).

La cara visible del sistema. Se comunica con la API REST (no con el
motor directamente), manteniendo la arquitectura cliente-servidor.

Arrancar (con la API ya corriendo en otra terminal):
    streamlit run interfaz/app.py
"""

import sys
import os

# Anadir la raiz del proyecto al path para encontrar config y otros modulos.
# Streamlit ejecuta desde la carpeta interfaz/, asi que sin esto no
# encontraria config.py ni el paquete interfaz.
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

import streamlit as st

from interfaz.cliente_api import ClienteAPI
from config import Config


# -- Configuracion de la pagina --

st.set_page_config(
    page_title="Asistente de Diabetes",
    page_icon="*",
    layout="centered",
)

# Cliente de la API (se crea una vez)
cliente = ClienteAPI(Config.API_BASE)


# -- Estado de la sesion --

def init_estado():
    """Inicializa las variables de sesion si no existen."""
    if "autenticado" not in st.session_state:
        st.session_state.autenticado = False
    if "codigo" not in st.session_state:
        st.session_state.codigo = ""
    if "nombre" not in st.session_state:
        st.session_state.nombre = ""
    if "rol" not in st.session_state:
        st.session_state.rol = ""
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []


# -- Pantalla de login --

def pantalla_login():
    """Muestra la pantalla de inicio de sesion."""
    st.title("Asistente Clinico de Diabetes")
    st.caption("Sistema conversacional con IA para informacion sobre diabetes mellitus")

    if not cliente.comprobar_conexion():
        st.error(
            "No se puede conectar con el servidor. "
            "Asegurate de que la API esta corriendo (python run_api.py)."
        )
        return

    st.markdown("---")
    st.subheader("Acceso")
    st.write("Introduce tu codigo de acceso para continuar.")

    codigo = st.text_input(
        "Codigo de acceso",
        type="password",
        placeholder="drab_...",
    )

    if st.button("Acceder", type="primary"):
        if not codigo:
            st.warning("Por favor, introduce un codigo.")
            return

        with st.spinner("Verificando..."):
            resultado = cliente.verificar_codigo(codigo)

        if resultado.get("valido"):
            st.session_state.autenticado = True
            st.session_state.codigo = codigo
            st.session_state.nombre = resultado["nombre"]
            st.session_state.rol = resultado["rol"]
            st.rerun()
        else:
            st.error("Codigo invalido o revocado. Intentalo de nuevo.")

    st.markdown("---")
    st.info(
        "Este asistente proporciona informacion general sobre diabetes "
        "basada en fuentes clinicas. No sustituye el consejo medico profesional."
    )


# -- Barra lateral --

def barra_lateral():
    """Muestra la barra lateral con info del usuario y navegacion."""
    with st.sidebar:
        st.markdown(f"### {st.session_state.nombre}")

        roles_display = {
            "lectura": "Paciente",
            "medico":  "Medico",
            "admin":   "Administrador",
        }
        st.markdown(f"**Rol:** {roles_display.get(st.session_state.rol, st.session_state.rol)}")

        st.markdown("---")

        if st.button("Cerrar sesion"):
            for clave in ["autenticado", "codigo", "nombre", "rol", "mensajes"]:
                st.session_state[clave] = False if clave == "autenticado" else (
                    [] if clave == "mensajes" else ""
                )
            st.rerun()

        if st.button("Limpiar conversacion"):
            st.session_state.mensajes = []
            st.rerun()

        st.markdown("---")
        st.caption("Sistema RAG Sanitario v1.0")
        st.caption(f"Modelo: {Config.LLM_MODEL}")


# -- Vista de chat --

def vista_chat():
    """Muestra el chat conversacional."""
    st.title("Consulta sobre diabetes")

    for msg in st.session_state.mensajes:
        with st.chat_message("user" if msg["rol"] == "usuario" else "assistant"):
            st.write(msg["texto"])
            if msg.get("fuentes"):
                with st.expander("Fuentes consultadas"):
                    for f in msg["fuentes"]:
                        st.caption(f"- {f}")
            if msg.get("pii_enmascarada", 0) > 0:
                st.caption(f"Se enmascararon {msg['pii_enmascarada']} dato(s) personal(es) por privacidad.")

    pregunta = st.chat_input("Escribe tu pregunta sobre diabetes...")

    if pregunta:
        st.session_state.mensajes.append({"rol": "usuario", "texto": pregunta})
        with st.chat_message("user"):
            st.write(pregunta)

        historial = [
            {"rol": m["rol"], "texto": m["texto"]}
            for m in st.session_state.mensajes[:-1]
        ]

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                resultado = cliente.preguntar(
                    pregunta,
                    st.session_state.codigo,
                    historial=historial,
                )

            if "error" in resultado:
                st.error(f"Error: {resultado['error']}")
            else:
                respuesta = resultado["respuesta"]
                st.write(respuesta)

                fuentes = resultado.get("fuentes", [])
                if fuentes and not resultado.get("bloqueado"):
                    with st.expander("Fuentes consultadas"):
                        for f in fuentes:
                            st.caption(f"- {f}")

                pii = resultado.get("pii_enmascarada", 0)
                if pii > 0:
                    st.caption(f"Se enmascararon {pii} dato(s) personal(es) por privacidad.")

                st.session_state.mensajes.append({
                    "rol":             "asistente",
                    "texto":           respuesta,
                    "fuentes":         fuentes,
                    "pii_enmascarada": pii,
                })


# -- Vista de documentos (medico+) --

def vista_documentos():
    """Muestra los documentos del corpus (solo medico y admin)."""
    st.title("Corpus documental")
    st.write("Documentos clinicos que el asistente usa como fuente.")

    with st.spinner("Cargando documentos..."):
        resultado = cliente.listar_documentos(st.session_state.codigo)

    if "error" in resultado:
        st.error(f"No se pudo cargar: {resultado['error']}")
        return

    col1, col2 = st.columns(2)
    col1.metric("Documentos", resultado.get("total_documentos", 0))
    col2.metric("Fragmentos indexados", resultado.get("total_fragmentos", 0))

    st.markdown("---")
    st.subheader("Lista de documentos")
    for doc in resultado.get("documentos", []):
        st.write(f"- {doc}")


# -- Vista de administracion (admin) --

def vista_admin():
    """Panel de administracion (solo admin)."""
    st.title("Administracion")

    tab1, tab2 = st.tabs(["Codigos de acceso", "Estadisticas"])

    with tab1:
        st.subheader("Crear nuevo codigo")
        col1, col2 = st.columns([2, 1])
        nombre_nuevo = col1.text_input("Nombre del usuario", key="nombre_nuevo")
        rol_nuevo = col2.selectbox("Rol", ["lectura", "medico", "admin"], key="rol_nuevo")

        if st.button("Crear codigo", type="primary"):
            if not nombre_nuevo:
                st.warning("Introduce un nombre.")
            else:
                resultado = cliente.crear_codigo(
                    st.session_state.codigo, nombre_nuevo, rol_nuevo
                )
                if "error" in resultado:
                    st.error(resultado["error"])
                else:
                    st.success("Codigo creado. Copialo ahora, no se mostrara de nuevo:")
                    st.code(resultado["codigo"], language=None)

        st.markdown("---")
        st.subheader("Codigos existentes")
        resultado = cliente.listar_codigos(st.session_state.codigo)
        if "error" in resultado:
            st.error(resultado["error"])
        else:
            for c in resultado.get("codigos", []):
                estado = "activo" if c["activa"] else "revocado"
                col1, col2, col3 = st.columns([2, 1, 1])
                col1.write(f"**{c['nombre']}** ({c['rol']})")
                col2.write(estado)
                col3.write(f"{c['total_usos']} usos")

    with tab2:
        st.subheader("Estadisticas de uso")
        resultado = cliente.estadisticas(st.session_state.codigo)
        if "error" in resultado:
            st.error(resultado["error"])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Codigos totales", resultado.get("total_codigos", 0))
            col2.metric("Codigos activos", resultado.get("codigos_activos", 0))
            col3.metric("Consultas totales", resultado.get("total_consultas", 0))


# -- Aplicacion principal --

def main():
    init_estado()

    if not st.session_state.autenticado:
        pantalla_login()
        return

    barra_lateral()

    rol = st.session_state.rol

    if rol == "admin":
        opciones = ["Chat", "Documentos", "Administracion"]
    elif rol == "medico":
        opciones = ["Chat", "Documentos"]
    else:
        opciones = ["Chat"]

    if len(opciones) > 1:
        seleccion = st.sidebar.radio("Navegacion", opciones)
    else:
        seleccion = opciones[0]

    if seleccion == "Chat":
        vista_chat()
    elif seleccion == "Documentos":
        vista_documentos()
    elif seleccion == "Administracion":
        vista_admin()


if __name__ == "__main__":
    main()
