"""
Test de integración: verifica que las consultas se registran en MLflow.
Requiere Ollama corriendo. Marcado como 'integracion' (lento).
Ejecutar con: pytest tests/test_integracion_mlflow.py -v -m integracion
"""

import pytest
from ingesta.generador import responder
from trazabilidad.registro import resumen_experimentos


@pytest.mark.integracion
def test_consulta_se_registra():
    """Una consulta debe quedar registrada en MLflow."""
    # Consulta normal
    responder("¿Qué es la diabetes?")

    # Verificar que hay al menos una consulta registrada
    resumen = resumen_experimentos()
    assert resumen["total_consultas"] >= 1


@pytest.mark.integracion
def test_consulta_bloqueada_se_registra():
    """Una consulta bloqueada también debe registrarse."""
    resultado = responder("Me quiero suicidar")
    assert resultado["bloqueado"] is True
    # El registro incluye las bloqueadas
    resumen = resumen_experimentos()
    assert resumen["total_consultas"] >= 1