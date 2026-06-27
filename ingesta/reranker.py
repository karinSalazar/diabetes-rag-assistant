"""
Re-ranker del sistema RAG.

Mejora la calidad de la recuperación con una segunda fase de filtrado.

Estrategia "retrieve-then-rerank":
  1. ChromaDB recupera muchos candidatos rápido (pero impreciso)
  2. El CrossEncoder los reordena por relevancia real (preciso)
  3. Se conservan solo los mejores

El CrossEncoder es más preciso que los embeddings porque lee la
pregunta y el fragmento JUNTOS, en lugar de compararlos por separado.

Uso:
    from ingesta.reranker import rerank
    mejores = rerank(pregunta, candidatos, top_k=4)
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import Config


# ── Carga del modelo (una sola vez) ──────────────────────────

@lru_cache(maxsize=1)
def _cargar_modelo() -> CrossEncoder:
    """
    Carga el modelo CrossEncoder. Usa lru_cache para cargarlo
    una única vez aunque se llame muchas veces (es costoso de cargar).
    """
    print("Cargando modelo de re-ranking (CrossEncoder)...")
    modelo = CrossEncoder(Config.RERANK_MODEL)
    print("Modelo de re-ranking listo.")
    return modelo


# ── Re-ranking ───────────────────────────────────────────────

def rerank(
    pregunta: str,
    candidatos: list[dict],
    top_k: int = None,
) -> list[dict]:
    """
    Reordena los candidatos por relevancia real respecto a la pregunta
    y devuelve los mejores.

    Args:
        pregunta:   la consulta del usuario
        candidatos: lista de dicts con al menos la clave "texto"
                    (los fragmentos recuperados por ChromaDB)
        top_k:      cuántos conservar (por defecto, Config.N_RESULTADOS_RAG)

    Returns:
        Lista de los top_k candidatos, reordenados de más a menos relevante.
        Cada dict conserva sus datos y se le añade "score_rerank".
    """
    k = top_k or Config.N_RESULTADOS_RAG

    # Si hay pocos candidatos, no hace falta reordenar
    if not candidatos:
        return []
    if len(candidatos) <= k:
        # Aun así calculamos el score para tener el dato
        pass

    modelo = _cargar_modelo()

    # El CrossEncoder evalúa pares (pregunta, fragmento)
    pares = [(pregunta, c["texto"]) for c in candidatos]
    scores = modelo.predict(pares)

    # Adjuntar la puntuación a cada candidato
    for candidato, score in zip(candidatos, scores):
        candidato["score_rerank"] = float(score)

    # Ordenar de mayor a menor puntuación y quedarse con los top_k
    ordenados = sorted(
        candidatos,
        key=lambda c: c["score_rerank"],
        reverse=True,
    )

    return ordenados[:k]


# ── Prueba directa del módulo ────────────────────────────────

if __name__ == "__main__":
    # Prueba con candidatos de ejemplo (sin necesitar ChromaDB)
    pregunta_prueba = "¿cuántas personas tienen diabetes?"

    candidatos_prueba = [
        {"texto": "La diabetes se trata con insulina y cambios en la dieta.",
         "fuente": "doc1"},
        {"texto": "Se estima que 346 millones de personas tienen diabetes en el mundo.",
         "fuente": "doc2"},
        {"texto": "El páncreas produce insulina para regular el azúcar.",
         "fuente": "doc3"},
        {"texto": "La prevalencia de diabetes ha aumentado en países de renta baja.",
         "fuente": "doc4"},
    ]

    print("=" * 60)
    print("PRUEBA DE RE-RANKING")
    print("=" * 60)
    print(f"\nPregunta: '{pregunta_prueba}'")
    print(f"\nCandidatos originales: {len(candidatos_prueba)}")

    mejores = rerank(pregunta_prueba, candidatos_prueba, top_k=2)

    print(f"\nMejores {len(mejores)} tras re-ranking:")
    for i, c in enumerate(mejores, 1):
        print(f"\n  [{i}] score: {c['score_rerank']:.4f} | fuente: {c['fuente']}")
        print(f"      {c['texto']}")

    print("\n(El fragmento con el dato '346 millones' debería quedar primero)")