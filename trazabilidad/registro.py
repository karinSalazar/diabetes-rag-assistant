"""
Trazabilidad del sistema RAG con MLflow.

Registra cada consulta del sistema para:
  1. Cumplimiento normativo (RGPD art. 30: registro de actividades
     de tratamiento; EU AI Act: trazabilidad de sistemas de IA sanitarios)
  2. Evaluación y análisis del rendimiento (Fase 9)

PRIVACIDAD: registra SIEMPRE la pregunta ya anonimizada, nunca la
original. De la PII detectada solo guarda el recuento, nunca los valores.
Así, ni el sistema de auditoría almacena datos personales.

Qué registra por consulta:
  - PARÁMETROS: modelo LLM, modelo embeddings, chunk_size, nº fragmentos,
                nivel de guardrail
  - MÉTRICAS:   tiempo de respuesta, nº fuentes, nº fragmentos,
                guardrail activado (0/1), nº PII detectada, longitud respuesta
  - ARTEFACTO:  ficha de texto con pregunta anonimizada, fuentes y respuesta

Uso:
    from trazabilidad.registro import inicializar_mlflow, registrar_consulta

    inicializar_mlflow()  # una vez al arrancar
    registrar_consulta(resultado, tiempo_seg=2.3)
"""

import os
import tempfile
from pathlib import Path

import mlflow

from config import Config


# ── Inicialización ───────────────────────────────────────────

_inicializado = False


def inicializar_mlflow():
    """
    Configura MLflow al arrancar la aplicación.
    Llamar una sola vez antes de registrar consultas.
    """
    global _inicializado
    if _inicializado:
        return

    # Dónde se guardan los registros (por defecto, carpeta local ./mlruns)
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)

    # Nombre del "experimento" que agrupa todas las consultas
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT)

    _inicializado = True
    print(f"MLflow inicializado → {Config.MLFLOW_TRACKING_URI}")
    print(f"Experimento: {Config.MLFLOW_EXPERIMENT}")


# ── Registro de una consulta ─────────────────────────────────

def registrar_consulta(resultado: dict, tiempo_seg: float):
    """
    Registra una consulta completa en MLflow.

    Args:
        resultado: el dict devuelto por la función responder() del generador.
                   Debe contener: pregunta_segura, respuesta, fuentes,
                   n_fragmentos, pii_detectada, bloqueado, categoria_bloqueo
        tiempo_seg: cuánto tardó la consulta en segundos

    Importante: usa resultado["pregunta_segura"] (anonimizada),
    NUNCA resultado["pregunta"] (que podría tener datos personales).
    """
    # Asegurar que MLflow está inicializado
    if not _inicializado:
        inicializar_mlflow()

    # Extraer datos del resultado de forma segura
    pregunta_segura  = resultado.get("pregunta_segura", "")
    respuesta        = resultado.get("respuesta", "")
    fuentes          = resultado.get("fuentes", [])
    n_fragmentos     = resultado.get("n_fragmentos", 0)
    pii_detectada    = resultado.get("pii_detectada", [])
    bloqueado        = resultado.get("bloqueado", False)
    categoria_bloq   = resultado.get("categoria_bloqueo", "ok")

    # Iniciar un "run" (una ficha de registro)
    with mlflow.start_run():

        # ── PARÁMETROS (configuración usada) ─────────────────
        mlflow.log_params({
            "modelo_llm":          Config.LLM_MODEL,
            "modelo_embedding":    Config.EMBEDDING_MODEL,
            "chunk_size":          Config.CHUNK_SIZE,
            "n_candidatos_rerank": Config.N_CANDIDATOS_RERANK,
            "n_resultados_rag":    Config.N_RESULTADOS_RAG,
            "categoria_resultado": categoria_bloq,
        })

        # ── MÉTRICAS (números medibles) ──────────────────────
        mlflow.log_metrics({
            "tiempo_respuesta_seg": round(tiempo_seg, 3),
            "n_fuentes_distintas":  len(set(fuentes)),
            "n_fragmentos":         n_fragmentos,
            "guardrail_activado":   1 if bloqueado else 0,
            "n_pii_detectada":      len(pii_detectada),
            "longitud_respuesta":   len(respuesta),
        })

        # ── ARTEFACTO (ficha de texto con el detalle) ────────
        # IMPORTANTE: solo datos anonimizados, sin PII
        tipos_pii = [d["tipo"] for d in pii_detectada]  # tipos, NO valores
        ficha = f"""REGISTRO DE CONSULTA
{'=' * 50}

Pregunta (anonimizada):
{pregunta_segura}

Estado: {'BLOQUEADO (' + categoria_bloq + ')' if bloqueado else 'Procesado correctamente'}

Tipos de datos personales detectados y enmascarados:
{', '.join(tipos_pii) if tipos_pii else 'Ninguno'}

Fuentes consultadas:
{chr(10).join('  - ' + f for f in fuentes) if fuentes else '  (ninguna)'}

Respuesta:
{respuesta}

Métricas:
  Tiempo de respuesta: {tiempo_seg:.3f} s
  Fragmentos usados:   {n_fragmentos}
  Fuentes distintas:   {len(set(fuentes))}
  PII enmascarada:     {len(pii_detectada)} elemento(s)
"""

        # Guardar la ficha como archivo temporal y subirla a MLflow
        ruta_tmp = Path(tempfile.gettempdir()) / "consulta_registro.txt"
        ruta_tmp.write_text(ficha, encoding="utf-8")
        mlflow.log_artifact(str(ruta_tmp), artifact_path="consultas")


# ── Consulta de estadísticas (para verificar) ────────────────

def resumen_experimentos() -> dict:
    """
    Devuelve un resumen de las consultas registradas.
    Útil para verificar que el registro funciona.
    """
    if not _inicializado:
        inicializar_mlflow()

    cliente = mlflow.tracking.MlflowClient()
    experimento = cliente.get_experiment_by_name(Config.MLFLOW_EXPERIMENT)

    if experimento is None:
        return {"total_consultas": 0}

    runs = cliente.search_runs(experiment_ids=[experimento.experiment_id])

    if not runs:
        return {"total_consultas": 0}

    tiempos = [r.data.metrics.get("tiempo_respuesta_seg", 0) for r in runs]
    bloqueos = sum(r.data.metrics.get("guardrail_activado", 0) for r in runs)
    pii_total = sum(r.data.metrics.get("n_pii_detectada", 0) for r in runs)

    return {
        "total_consultas":   len(runs),
        "tiempo_medio_seg":  round(sum(tiempos) / len(tiempos), 3) if tiempos else 0,
        "tiempo_max_seg":    round(max(tiempos), 3) if tiempos else 0,
        "consultas_bloqueadas": int(bloqueos),
        "pii_total_enmascarada": int(pii_total),
    }


# ── Prueba directa del módulo ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DEL MÓDULO DE TRAZABILIDAD MLflow")
    print("=" * 60)

    inicializar_mlflow()

    # Simular el registro de una consulta de ejemplo
    resultado_ejemplo = {
        "pregunta":          "Soy Ana, mi DNI es 11111111H, ¿qué es la diabetes?",
        "pregunta_segura":   "Soy [NOMBRE], mi DNI es [DNI], ¿qué es la diabetes?",
        "respuesta":         "La diabetes es una enfermedad crónica que afecta los niveles de glucosa en sangre.",
        "fuentes":           ["who_factsheet_diabetes.pdf", "who_diabetes_factsheet.html"],
        "n_fragmentos":      4,
        "pii_detectada":     [{"tipo": "NOMBRE", "valor": "Ana"}, {"tipo": "DNI", "valor": "11111111H"}],
        "bloqueado":         False,
        "categoria_bloqueo": "ok",
    }

    print("\nRegistrando una consulta de ejemplo...")
    registrar_consulta(resultado_ejemplo, tiempo_seg=2.34)
    print("Consulta registrada.")

    print("\nResumen de experimentos registrados:")
    resumen = resumen_experimentos()
    for clave, valor in resumen.items():
        print(f"  {clave}: {valor}")

    print("\nPara ver el panel visual de MLflow, ejecuta en otra terminal:")
    print("  mlflow ui")
    print("Y abre http://localhost:5000 en tu navegador.")