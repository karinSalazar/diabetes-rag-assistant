"""
Anonimizador de datos personales (PII) del sistema RAG.

Detecta y enmascara información personal identificable ANTES de que
el texto llegue al modelo de lenguaje o se registre en logs.

Cumple el principio de minimización de datos del RGPD (art. 5.1.c):
solo se procesa la información estrictamente necesaria (el dato clínico),
enmascarando todo dato identificativo de la persona.

Combina dos motores de detección:
  1. Regex propios afinados a formatos españoles (DNI, IBAN, teléfono...)
  2. Presidio + spaCy para nombres de persona y direcciones (NER en español)

IMPORTANTE: conserva los datos clínicos (HbA1c, glucemia, medicación),
ya que NO son datos identificativos y son necesarios para responder.

Uso:
    from privacidad.anonimizador import anonimizar

    resultado = anonimizar("Soy María, DNI 12345678X, mi HbA1c es 8.5")
    print(resultado["texto_anonimizado"])
    # → "Soy [NOMBRE], DNI [DNI], mi HbA1c es 8.5"
"""

import re

import spacy

from config import Config


# ── Patrones regex para formatos españoles ───────────────────
# Cada patrón detecta un tipo de dato personal con formato fijo.
# El orden importa: se aplican de más específico a más general.

PATRONES = [
    # DNI español: 8 dígitos + letra (ej. 12345678X)
    ("DNI", re.compile(r"\b\d{8}[-\s]?[A-Za-z]\b")),

    # NIE: letra + 7 dígitos + letra (ej. X1234567L)
    ("NIE", re.compile(r"\b[XYZxyz][-\s]?\d{7}[-\s]?[A-Za-z]\b")),

    # IBAN español: ES + 22 dígitos (puede llevar espacios)
    ("IBAN", re.compile(r"\bES\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}\b")),

    # Número Seguridad Social: 11-12 dígitos (puede llevar separadores)
    ("NUM_SS", re.compile(r"\b\d{2}[-\s/]?\d{8,10}[-\s/]?\d{0,2}\b")),

    # Número de historia clínica (patrones comunes: HC/NHC + dígitos)
    ("HISTORIA_CLINICA", re.compile(r"\b(?:HC|NHC|HISTORIA)[-\s:]*\d{4,12}\b", re.IGNORECASE)),

    # Email
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),

    # Teléfono español: 9 dígitos empezando por 6,7,8,9 (con prefijo opcional +34)
    ("TELEFONO", re.compile(r"\b(?:\+34[-\s]?)?[6789]\d{2}[-\s]?\d{3}[-\s]?\d{3}\b")),

    # Fecha de nacimiento (formatos dd/mm/aaaa, dd-mm-aaaa)
    ("FECHA_NAC", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),

    # Código postal español (5 dígitos) — opcional, puede dar falsos positivos
    # Lo dejamos comentado por defecto para no enmascarar números clínicos
    # ("CODIGO_POSTAL", re.compile(r"\b\d{5}\b")),
]


# ── Carga del modelo NER de spaCy ────────────────────────────

_nlp = None

def _cargar_nlp():
    """Carga el modelo de spaCy en español (una sola vez)."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(Config.PII_SPACY_MODEL)
    return _nlp


# ── Detección con regex ──────────────────────────────────────

def _enmascarar_regex(texto: str) -> tuple[str, list[dict]]:
    """
    Aplica los patrones regex y enmascara las coincidencias.
    Devuelve el texto enmascarado y la lista de entidades detectadas.
    """
    detecciones = []
    resultado = texto

    for etiqueta, patron in PATRONES:
        for match in patron.finditer(resultado):
            detecciones.append({
                "tipo":  etiqueta,
                "valor": match.group(),
                "metodo": "regex",
            })
        # Reemplazar todas las coincidencias por la etiqueta
        resultado = patron.sub(f"[{etiqueta}]", resultado)

    return resultado, detecciones


# ── Detección con NER (nombres y lugares) ────────────────────

def _enmascarar_ner(texto: str) -> tuple[str, list[dict]]:
    """
    Usa spaCy para detectar nombres de personas y lugares,
    que no tienen formato fijo.

    Etiquetas de spaCy en español:
      PER = persona, LOC = lugar, ORG = organización
    """
    nlp = _cargar_nlp()
    doc = nlp(texto)

    detecciones = []
    entidades = []
    for ent in doc.ents:
        # Ignorar entidades que en realidad son etiquetas ya puestas
        # por el regex (van entre corchetes, ej. [DNI], [HISTORIA_CLINICA])
        contexto_anterior = texto[max(0, ent.start_char - 1):ent.start_char]
        contexto_posterior = texto[ent.end_char:ent.end_char + 1]
        if contexto_anterior == "[" or contexto_posterior == "]":
            continue
        # Ignorar también si el texto de la entidad está todo en mayúsculas
        # con guiones bajos (patrón típico de nuestras etiquetas)
        if "_" in ent.text and ent.text.isupper():
            continue

        if ent.label_ == "PER":
            entidades.append((ent.start_char, ent.end_char, "NOMBRE", ent.text))
        elif ent.label_ == "LOC":
            entidades.append((ent.start_char, ent.end_char, "DIRECCION", ent.text))

    entidades.sort(key=lambda e: e[0], reverse=True)

    resultado = texto
    for inicio, fin, etiqueta, valor in entidades:
        detecciones.append({
            "tipo":   etiqueta,
            "valor":  valor,
            "metodo": "ner",
        })
        resultado = resultado[:inicio] + f"[{etiqueta}]" + resultado[fin:]

    return resultado, detecciones


# ── Función principal ────────────────────────────────────────

def anonimizar(texto: str) -> dict:
    """
    Anonimiza un texto detectando y enmascarando datos personales.

    Proceso:
      1. Primero regex (formatos fijos: DNI, teléfono, email, IBAN...)
      2. Luego NER (nombres de persona, direcciones)

    Args:
        texto: el texto original que puede contener datos personales

    Returns:
        dict con:
          - texto_original:    el texto tal cual entró
          - texto_anonimizado: el texto con los datos enmascarados
          - detecciones:       lista de datos personales encontrados
          - hubo_pii:          True si se detectó algún dato personal
    """
    if not texto or not texto.strip():
        return {
            "texto_original":    texto,
            "texto_anonimizado": texto,
            "detecciones":       [],
            "hubo_pii":          False,
        }

    # 1. Regex primero (los formatos fijos son más fiables)
    paso1, det_regex = _enmascarar_regex(texto)

    # 2. NER después (nombres y lugares sobre el texto ya parcialmente limpio)
    paso2, det_ner = _enmascarar_ner(paso1)

    todas_detecciones = det_regex + det_ner

    return {
        "texto_original":    texto,
        "texto_anonimizado": paso2,
        "detecciones":       todas_detecciones,
        "hubo_pii":          len(todas_detecciones) > 0,
    }


# ── Prueba directa del módulo ────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBA DEL ANONIMIZADOR DE PII")
    print("=" * 60)

    casos_prueba = [
        # Caso 1: consulta con muchos datos personales
        "Hola, soy María González, mi DNI es 12345678X y mi teléfono "
        "es 612345678. Mi HbA1c salió en 8.5 y mi médico me cambió la metformina.",

        # Caso 2: email e IBAN
        "Mi correo es maria.gonzalez@gmail.com y mi cuenta es "
        "ES9121000418450200051332 para el pago de la consulta.",

        # Caso 3: solo dato clínico (NO debe enmascarar nada)
        "¿Cuál es el valor normal de glucosa en ayunas?",

        # Caso 4: historia clínica y fecha de nacimiento
        "Mi número de historia clínica es NHC 4567890 y nací el 15/03/1985.",
    ]

    for i, caso in enumerate(casos_prueba, 1):
        resultado = anonimizar(caso)
        print(f"\n{'─'*60}")
        print(f"CASO {i}:")
        print(f"  Original:    {resultado['texto_original']}")
        print(f"  Anonimizado: {resultado['texto_anonimizado']}")
        if resultado["detecciones"]:
            print(f"  Detectado:")
            for d in resultado["detecciones"]:
                print(f"    - {d['tipo']}: '{d['valor']}' (vía {d['metodo']})")
        else:
            print(f"  Sin datos personales (correcto si es solo clínico)")