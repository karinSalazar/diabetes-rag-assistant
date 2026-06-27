"""
Indexador del sistema RAG.

Toma los fragmentos extraídos por el cargador, genera el embedding
(vector de significado) de cada uno usando Ollama, y los almacena
en ChromaDB para permitir la búsqueda semántica.

Flujo:
    documentos → cargador → fragmentos → indexador → ChromaDB

Uso:
    # Indexar toda la carpeta data/raw
    python -m ingesta.indexador

    # O desde código:
    from ingesta.indexador import indexar_carpeta, buscar
    indexar_carpeta("data/raw")
    resultados = buscar("¿qué es la hipoglucemia?")
"""

# Desactivar telemetría de ChromaDB ANTES de importarlo
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

from pathlib import Path

import chromadb
import ollama

from config import Config
from ingesta.cargador import cargar_carpeta, Fragmento


# Cliente de Ollama (apunta al servidor local)
_ollama = ollama.Client(host=Config.OLLAMA_HOST)


# ── Generación de embeddings ─────────────────────────────────

def generar_embedding(texto: str) -> list[float]:
    """
    Convierte un texto en su vector de significado (embedding)
    usando el modelo nomic-embed-text de Ollama.

    Returns:
        Lista de floats (el vector). Para nomic-embed-text son 768 dimensiones.
    """
    respuesta = _ollama.embeddings(
        model=Config.EMBEDDING_MODEL,
        prompt=texto,
    )
    return respuesta["embedding"]


# ── Limpieza de metadatos para ChromaDB ──────────────────────

def _limpiar_metadata(metadata: dict) -> dict:
    """
    ChromaDB no acepta valores None en los metadatos.
    Convierte None a cadena vacía y deja el resto igual.
    """
    limpia = {}
    for clave, valor in metadata.items():
        if valor is None:
            limpia[clave] = ""
        elif isinstance(valor, (str, int, float, bool)):
            limpia[clave] = valor
        else:
            limpia[clave] = str(valor)
    return limpia


# ── Conexión a ChromaDB ──────────────────────────────────────

def _obtener_coleccion(reiniciar: bool = False):
    """
    Conecta con ChromaDB y devuelve la colección de documentos.

    Args:
        reiniciar: si True, borra la colección existente y crea una nueva.
                   Útil para reindexar desde cero sin duplicados.
    """
    # PersistentClient guarda los datos en disco (no se pierden al cerrar)
    Path(Config.VECTORDB_PATH).mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=Config.VECTORDB_PATH)

    if reiniciar:
        try:
            cliente.delete_collection(Config.COLLECTION_NAME)
            print("Colección anterior eliminada (reindexando desde cero).")
        except Exception:
            pass  # no existía, no pasa nada

    # get_or_create: la usa si existe, la crea si no.
    # metadata hnsw:space=cosine → usa distancia coseno para la similitud
    coleccion = cliente.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return coleccion


# ── Indexación ───────────────────────────────────────────────

def indexar_fragmentos(fragmentos: list[Fragmento], reiniciar: bool = True) -> int:
    """
    Indexa una lista de fragmentos en ChromaDB.

    Args:
        fragmentos: lista de Fragmento (del cargador)
        reiniciar:  si True, reconstruye la colección desde cero

    Returns:
        Número total de fragmentos indexados en la colección.
    """
    if not fragmentos:
        print("No hay fragmentos que indexar.")
        return 0

    coleccion = _obtener_coleccion(reiniciar=reiniciar)

    total = len(fragmentos)
    print(f"Generando embeddings e indexando {total} fragmentos...")
    print("(esto tarda unos minutos en CPU, es normal)\n")

    # Procesamos en lotes para mostrar progreso y no saturar memoria
    LOTE = 16
    for inicio in range(0, total, LOTE):
        lote = fragmentos[inicio:inicio + LOTE]

        # IDs únicos para cada fragmento
        ids        = [f"frag_{inicio + j:06d}" for j in range(len(lote))]
        textos     = [f.texto for f in lote]
        metadatas  = [_limpiar_metadata(f.metadata) for f in lote]
        embeddings = [generar_embedding(t) for t in textos]

        coleccion.add(
            ids=ids,
            documents=textos,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        procesados = min(inicio + LOTE, total)
        porcentaje = procesados * 100 // total
        print(f"  {procesados}/{total} fragmentos ({porcentaje}%)", end="\r")

    print()  # salto de línea tras la barra de progreso
    total_indexado = coleccion.count()
    print(f"\nIndexación completa: {total_indexado} fragmentos en ChromaDB.")
    return total_indexado


def indexar_carpeta(ruta_carpeta: str = "data/raw") -> int:
    """
    Carga e indexa todos los documentos de una carpeta de una sola vez.
    Combina cargador + indexador.
    """
    fragmentos = cargar_carpeta(ruta_carpeta)
    print()  # separación visual
    return indexar_fragmentos(fragmentos, reiniciar=True)


# ── Búsqueda (para probar la recuperación) ───────────────────

def buscar(pregunta: str, n_resultados: int = 3) -> list[dict]:
    """
    Busca los fragmentos más relevantes para una pregunta.
    Sirve para comprobar que la recuperación funciona.

    Args:
        pregunta:     la consulta en lenguaje natural
        n_resultados: cuántos fragmentos devolver

    Returns:
        Lista de dicts con texto, fuente, página y distancia.
    """
    coleccion = _obtener_coleccion(reiniciar=False)

    if coleccion.count() == 0:
        print("La colección está vacía. Indexa documentos primero.")
        return []

    # Generar el embedding de la pregunta y buscar los más cercanos
    emb_pregunta = generar_embedding(pregunta)
    resultados = coleccion.query(
        query_embeddings=[emb_pregunta],
        n_results=n_resultados,
    )

    salida = []
    documentos = resultados["documents"][0]
    metadatas  = resultados["metadatas"][0]
    distancias = resultados["distances"][0]

    for doc, meta, dist in zip(documentos, metadatas, distancias):
        salida.append({
            "texto":     doc,
            "fuente":    meta.get("fuente", "?"),
            "pagina":    meta.get("pagina", ""),
            "distancia": round(dist, 4),
        })
    return salida


# ── Prueba directa del módulo ────────────────────────────────

if __name__ == "__main__":
    # 1. Indexar la carpeta data/raw
    indexar_carpeta("data/raw")

    # 2. Probar algunas búsquedas para ver que la recuperación funciona
    print("\n" + "=" * 60)
    print("PRUEBA DE BÚSQUEDA SEMÁNTICA")
    print("=" * 60)

    preguntas_prueba = [
        "¿qué es la diabetes?",
        "how many people have diabetes in the world?",
        "prevención de la diabetes tipo 2",
    ]

    for pregunta in preguntas_prueba:
        print(f"\nPregunta: '{pregunta}'")
        resultados = buscar(pregunta, n_resultados=2)
        for i, r in enumerate(resultados, 1):
            pagina = f" (pág. {r['pagina']})" if r["pagina"] else ""
            print(f"  [{i}] {r['fuente']}{pagina} — distancia: {r['distancia']}")
            print(f"      {r['texto'][:150]}...")
