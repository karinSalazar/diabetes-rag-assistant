"""
Generador de respuestas del sistema RAG.

Este módulo completa el ciclo RAG (Retrieval-Augmented Generation):
  1. RECUPERA los fragmentos relevantes de ChromaDB (Retrieval)
  2. GENERA una respuesta con Llama 3.2 basada en ellos (Generation)

El modelo solo puede responder con la información de los documentos
recuperados. Si la respuesta no está en ellos, lo dice claramente
en lugar de inventar (evita alucinaciones).

Uso:
    from ingesta.generador import responder

    resultado = responder("¿qué es la diabetes?")
    print(resultado["respuesta"])
    print(resultado["fuentes"])
"""

# Desactivar telemetría de ChromaDB antes de importarlo
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

import ollama

from config import Config
from ingesta.indexador import buscar, _obtener_coleccion, generar_embedding
from ingesta.reranker import rerank
from privacidad.anonimizador import anonimizar
from privacidad.guardrails import validar_entrada, validar_salida


# Cliente de Ollama
_ollama = ollama.Client(host=Config.OLLAMA_HOST)


# ── Instrucciones para el modelo (prompt del sistema) ────────
# Estas reglas son lo que convierte a Llama en un asistente clínico
# seguro: solo usa el contexto, no inventa, cita fuentes.

PROMPT_SISTEMA = """Eres un asistente clínico especializado en diabetes mellitus.

Tu función es ayudar a entender información sobre diabetes basándote
ÚNICAMENTE en el contexto médico que se te proporciona.

REGLAS IMPORTANTES:
- Responde SIEMPRE en español claro y comprensible.
- Usa SOLO la información del contexto proporcionado.
- Si la información no está en el contexto, di exactamente:
  "No tengo información suficiente en mi base clínica para responder eso
  con seguridad. Te recomiendo consultar con tu médico."
- NUNCA inventes datos, dosis, valores de referencia ni tratamientos.
- Para cualquier decisión clínica concreta, recomienda consultar
  con un profesional sanitario.
- Sé claro y empático, pero riguroso.
"""


# ── Construcción del contexto ────────────────────────────────

def _construir_contexto(fragmentos: list[dict]) -> tuple[str, list[str]]:
    """
    Junta los fragmentos recuperados en un único texto de contexto
    para pasárselo al modelo, e identifica las fuentes usadas.

    Returns:
        (contexto, lista_de_fuentes)
    """
    bloques = []
    fuentes = set()

    for frag in fragmentos:
        fuente = frag.get("fuente", "documento")
        pagina = frag.get("pagina", "")
        texto  = frag.get("texto", "")

        # Etiqueta de origen para que el modelo sepa de dónde viene
        etiqueta = f"[Fuente: {fuente}"
        if pagina:
            etiqueta += f", página {pagina}"
        etiqueta += "]"

        bloques.append(f"{etiqueta}\n{texto}")
        fuentes.add(fuente)

    contexto = "\n\n---\n\n".join(bloques)
    return contexto, sorted(fuentes)


# ── Función principal: responder una pregunta ────────────────

def responder(
    pregunta: str,
    n_fragmentos: int = None,
    historial: list[dict] = None,
) -> dict:
    """
    Responde una pregunta usando RAG completo.

    Args:
        pregunta:     la consulta del usuario en lenguaje natural
        n_fragmentos: cuántos fragmentos recuperar (por defecto, Config)
        historial:    lista de mensajes previos para memoria de conversación
                      formato: [{"rol": "usuario"/"asistente", "texto": "..."}]

    Returns:
        dict con:
          - pregunta:     la pregunta original
          - respuesta:    el texto generado por el modelo
          - fuentes:      lista de documentos usados
          - n_fragmentos: cuántos fragmentos se recuperaron
    """
    n = n_fragmentos or Config.N_RESULTADOS_RAG
    historial = historial or []

    # ── ANONIMIZACIÓN: enmascarar datos personales ANTES de procesar ──
    # El texto que llega al modelo y a los logs nunca contiene PII.
    resultado_anon = anonimizar(pregunta)
    pregunta_segura = resultado_anon["texto_anonimizado"]
    pii_detectada = resultado_anon["detecciones"]

    # ── GUARDRAIL DE ENTRADA: validar la pregunta antes de procesarla ──
    validacion_entrada = validar_entrada(pregunta_segura, nivel="equilibrado")
    if not validacion_entrada.es_valida:
        # La pregunta fue bloqueada: devolver el mensaje de seguridad
        return {
            "pregunta":         pregunta,
            "pregunta_segura":  pregunta_segura,
            "respuesta":        validacion_entrada.mensaje,
            "fuentes":          [],
            "n_fragmentos":     0,
            "pii_detectada":    pii_detectada,
            "bloqueado":        True,
            "categoria_bloqueo": validacion_entrada.categoria,
        }
    
    # ── 1. RECUPERAR: buscar fragmentos relevantes en ChromaDB ──
    coleccion = _obtener_coleccion(reiniciar=False)
    if coleccion.count() == 0:
        return {
            "pregunta":     pregunta,
            "respuesta":    "La base clínica está vacía. Indexa documentos primero.",
            "fuentes":      [],
            "n_fragmentos": 0,
        }

    # Recuperar MÁS candidatos de los necesarios (para que el reranker elija)
    n_candidatos = Config.N_CANDIDATOS_RERANK
    emb_pregunta = generar_embedding(pregunta_segura)
    resultados = coleccion.query(
        query_embeddings=[emb_pregunta],
        n_results=min(n_candidatos, coleccion.count()),
    )

    # Empaquetar los candidatos recuperados
    candidatos = []
    for doc, meta in zip(resultados["documents"][0], resultados["metadatas"][0]):
        candidatos.append({
            "texto":  doc,
            "fuente": meta.get("fuente", "?"),
            "pagina": meta.get("pagina", ""),
        })

    # RE-RANKING: reordenar por relevancia real y quedarse con los mejores
    fragmentos = rerank(pregunta_segura, candidatos, top_k=n)
    contexto, fuentes = _construir_contexto(fragmentos)

    # ── 2. GENERAR: construir los mensajes para Llama ───────────
    mensajes = [{"role": "system", "content": PROMPT_SISTEMA}]

    # Añadir historial previo (memoria de conversación)
    for msg in historial:
        rol = "user" if msg["rol"] == "usuario" else "assistant"
        mensajes.append({"role": rol, "content": msg["texto"]})

    # El mensaje actual: contexto + pregunta
    mensaje_usuario = (
        f"Contexto médico:\n{contexto}\n\n"
        f"Pregunta del usuario: {pregunta_segura}\n\n"
        f"Responde basándote únicamente en el contexto anterior."
    )
    mensajes.append({"role": "user", "content": mensaje_usuario})

    # Llamar a Llama 3.2 para generar la respuesta
    respuesta_llm = _ollama.chat(
        model=Config.LLM_MODEL,
        messages=mensajes,
    )
    texto_respuesta = respuesta_llm["message"]["content"].strip()

    # ── GUARDRAIL DE SALIDA: validar la respuesta antes de mostrarla ──
    validacion_salida = validar_salida(texto_respuesta)
    if not validacion_salida.es_valida:
        texto_respuesta = validacion_salida.mensaje

    return {
        "pregunta":          pregunta,           # la original (para mostrar al usuario)
        "pregunta_segura":   pregunta_segura,    # la anonimizada (la que vio el modelo)
        "respuesta":         texto_respuesta,
        "fuentes":           fuentes,
        "n_fragmentos":      len(fragmentos),
        "pii_detectada":     pii_detectada,      # qué datos personales se enmascararon
        "bloqueado":         False,
        "categoria_bloqueo": "ok",
    }


# ── Prueba directa del módulo ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DEL GENERADOR DE RESPUESTAS (RAG completo)")
    print("=" * 60)
    print("(cada respuesta tarda un poco porque Llama redacta en CPU)\n")

    preguntas = [
        "¿Qué es la diabetes?",
        "¿Cuántas personas tienen diabetes en el mundo?",
        "¿Cómo se puede prevenir la diabetes tipo 2?",
        "¿Cuál es la dosis de insulina para un niño de 5 años?",
    ]

    for pregunta in preguntas:
        print("\n" + "─" * 60)
        print(f"PREGUNTA: {pregunta}")
        print("─" * 60)

        resultado = responder(pregunta)

        print(f"\nRESPUESTA:\n{resultado['respuesta']}")
        print(f"\nFUENTES: {', '.join(resultado['fuentes'])}")
        print(f"(basado en {resultado['n_fragmentos']} fragmentos)")