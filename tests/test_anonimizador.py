"""
Tests de la capa de anonimización PII.
Ejecutar con: pytest tests/test_anonimizador.py -v
"""

from privacidad.anonimizador import anonimizar


def test_detecta_dni():
    """El DNI español debe detectarse y enmascararse."""
    resultado = anonimizar("Mi DNI es 12345678X")
    assert "[DNI]" in resultado["texto_anonimizado"]
    assert "12345678X" not in resultado["texto_anonimizado"]


def test_detecta_telefono():
    """El teléfono español debe detectarse."""
    resultado = anonimizar("Llámame al 612345678")
    assert "[TELEFONO]" in resultado["texto_anonimizado"]


def test_detecta_email():
    """El email debe detectarse."""
    resultado = anonimizar("Mi correo es test@example.com")
    assert "[EMAIL]" in resultado["texto_anonimizado"]


def test_conserva_dato_clinico():
    """CRÍTICO: el dato clínico NO debe enmascararse."""
    resultado = anonimizar("Mi HbA1c es 8.5 y tomo metformina")
    assert "8.5" in resultado["texto_anonimizado"]
    assert "metformina" in resultado["texto_anonimizado"]


def test_pregunta_sin_pii_no_cambia():
    """Una pregunta clínica sin datos personales no debe alterarse."""
    texto = "¿Cuál es el valor normal de glucosa?"
    resultado = anonimizar(texto)
    assert resultado["texto_anonimizado"] == texto
    assert resultado["hubo_pii"] is False