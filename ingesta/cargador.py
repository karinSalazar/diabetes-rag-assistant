"""
Cargador de documentos del sistema RAG.

Lee documentos clínicos en varios formatos (PDF, CSV, Excel, HTML, TXT),
extrae su texto y lo divide en fragmentos (chunks) con metadatos.

Cada fragmento conserva información de su origen (documento, página, hoja),
lo que permite al chatbot citar la fuente exacta de cada respuesta.

Uso:
    from ingesta.cargador import cargar_documento, cargar_carpeta

    # Un solo documento
    fragmentos = cargar_documento("data/raw/who_factsheet.pdf")

    # Toda una carpeta
    fragmentos = cargar_carpeta("data/raw")
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import warnings
warnings.filterwarnings("ignore", message=".*ARC4.*")

from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
from pypdf import PdfReader
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Config


# ── Estructura de un fragmento ───────────────────────────────

@dataclass
class Fragmento:
    """
    Un fragmento de texto extraído de un documento, con sus metadatos.

    Atributos:
        texto:    el contenido textual del fragmento
        metadata: diccionario con origen (fuente, formato, página, hoja...)
    """
    texto:    str
    metadata: dict = field(default_factory=dict)


# ── Lectores por formato ─────────────────────────────────────
# Cada función lee un formato y devuelve una lista de "secciones".
# Una sección es un bloque de texto con su ubicación (página/hoja).

def _leer_pdf(ruta: Path) -> list[dict]:
    """Lee un PDF página por página, conservando el número de página."""
    lector = PdfReader(str(ruta))
    secciones = []
    for i, pagina in enumerate(lector.pages):
        texto = pagina.extract_text()
        if texto and texto.strip():
            secciones.append({
                "texto":         texto.strip(),
                "pagina":        i + 1,
                "total_paginas": len(lector.pages),
                "hoja":          None,
            })
    return secciones


def _leer_csv(ruta: Path) -> list[dict]:
    """
    Lee un CSV. Para no crear fragmentos demasiado pequeños,
    agrupa las filas en bloques de 50.
    """
    df = pd.read_csv(ruta)
    columnas = ", ".join(df.columns.astype(str).tolist())

    BLOQUE = 50
    secciones = []
    for inicio in range(0, len(df), BLOQUE):
        bloque = df.iloc[inicio:inicio + BLOQUE]
        lineas = [f"Columnas: {columnas}"]
        for _, fila in bloque.iterrows():
            pares = [f"{c}: {v}" for c, v in fila.items() if pd.notna(v)]
            lineas.append(" | ".join(pares))
        secciones.append({
            "texto":         "\n".join(lineas),
            "pagina":        None,
            "total_paginas": None,
            "hoja":          f"filas_{inicio + 1}_a_{min(inicio + BLOQUE, len(df))}",
        })
    return secciones


def _leer_excel(ruta: Path) -> list[dict]:
    """Lee un Excel hoja por hoja, conservando el nombre de la hoja."""
    libro = pd.ExcelFile(ruta)
    secciones = []
    for hoja in libro.sheet_names:
        df = libro.parse(hoja)
        if df.empty:
            continue
        columnas = ", ".join(df.columns.astype(str).tolist())
        lineas = [f"Hoja: {hoja}", f"Columnas: {columnas}"]
        for _, fila in df.iterrows():
            pares = [f"{c}: {v}" for c, v in fila.items() if pd.notna(v)]
            if pares:
                lineas.append(" | ".join(pares))
        secciones.append({
            "texto":         "\n".join(lineas),
            "pagina":        None,
            "total_paginas": len(libro.sheet_names),
            "hoja":          hoja,
        })
    return secciones


def _leer_html(ruta: Path) -> list[dict]:
    """Lee un HTML y extrae solo el texto legible, sin etiquetas ni scripts."""
    contenido = ruta.read_text(encoding="utf-8", errors="ignore")
    sopa = BeautifulSoup(contenido, "lxml")

    # Eliminar elementos que no son contenido (scripts, menús, etc.)
    for etiqueta in sopa(["script", "style", "nav", "footer", "header", "aside"]):
        etiqueta.decompose()

    # Líneas de navegación web que no son contenido clínico
    BASURA = {
        "skip to main content", "credits", "menu", "search",
        "home", "close", "share", "print", "download",
        "subscribe", "newsletter", "cookies", "accept",
    }

    lineas = []
    for linea in sopa.get_text("\n").splitlines():
        limpia = linea.strip()
        # Saltar líneas vacías, muy cortas, o que son basura de navegación
        if not limpia or len(limpia) < 3:
            continue
        if limpia.lower() in BASURA:
            continue
        lineas.append(limpia)

    texto = "\n".join(lineas)

    return [{
        "texto":         texto,
        "pagina":        None,
        "total_paginas": None,
        "hoja":          None,
    }]


def _leer_txt(ruta: Path) -> list[dict]:
    """Lee un archivo de texto plano."""
    texto = ruta.read_text(encoding="utf-8", errors="ignore")
    return [{
        "texto":         texto.strip(),
        "pagina":        None,
        "total_paginas": None,
        "hoja":          None,
    }]


# Mapa de extensión → función lectora
_LECTORES = {
    ".pdf":  _leer_pdf,
    ".csv":  _leer_csv,
    ".xlsx": _leer_excel,
    ".xls":  _leer_excel,
    ".html": _leer_html,
    ".htm":  _leer_html,
    ".txt":  _leer_txt,
}

FORMATOS_SOPORTADOS = set(_LECTORES.keys())


# ── Función principal: cargar un documento ───────────────────

def cargar_documento(ruta_archivo: str | Path) -> list[Fragmento]:
    """
    Carga un documento, extrae su texto y lo divide en fragmentos.

    Args:
        ruta_archivo: ruta al documento (PDF, CSV, Excel, HTML o TXT)

    Returns:
        Lista de Fragmento, cada uno con su texto y metadatos.

    Lanza:
        ValueError si el formato no está soportado.
        FileNotFoundError si el archivo no existe.
    """
    ruta = Path(ruta_archivo)

    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    extension = ruta.suffix.lower()
    if extension not in _LECTORES:
        raise ValueError(
            f"Formato '{extension}' no soportado. "
            f"Usa uno de: {', '.join(sorted(FORMATOS_SOPORTADOS))}"
        )

    # 1. Leer el documento según su formato
    lector = _LECTORES[extension]
    secciones = lector(ruta)

    if not secciones:
        print(f"  Aviso: no se extrajo texto de {ruta.name}")
        return []

    # 2. Preparar el divisor de texto (chunking)
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # 3. Trocear cada sección y añadir metadatos
    fragmentos = []
    for seccion in secciones:
        # Saltar secciones con muy poco texto
        if len(seccion["texto"].strip()) < 30:
            continue

        trozos = divisor.split_text(seccion["texto"])
        for trozo in trozos:
            fragmentos.append(Fragmento(
                texto=trozo,
                metadata={
                    "fuente":        ruta.name,
                    "formato":       extension.lstrip("."),
                    "pagina":        seccion["pagina"],
                    "hoja":          seccion["hoja"],
                    "total_paginas": seccion["total_paginas"],
                    "especialidad":  "diabetes",
                },
            ))

    return fragmentos


# ── Cargar una carpeta entera ────────────────────────────────

def cargar_carpeta(ruta_carpeta: str | Path) -> list[Fragmento]:
    """
    Carga todos los documentos soportados de una carpeta.

    Args:
        ruta_carpeta: carpeta con los documentos (ej. "data/raw")

    Returns:
        Lista con TODOS los fragmentos de todos los documentos.
    """
    carpeta = Path(ruta_carpeta)

    if not carpeta.exists():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")

    archivos = [
        f for f in carpeta.iterdir()
        if f.is_file() and f.suffix.lower() in FORMATOS_SOPORTADOS
    ]

    if not archivos:
        print(f"No se encontraron documentos soportados en {carpeta}")
        return []

    print(f"Cargando {len(archivos)} documento(s) de {carpeta}...\n")

    todos_fragmentos = []
    for archivo in sorted(archivos):
        fragmentos = cargar_documento(archivo)
        todos_fragmentos.extend(fragmentos)
        print(f"  {archivo.name}: {len(fragmentos)} fragmentos")

    print(f"\nTotal: {len(todos_fragmentos)} fragmentos de {len(archivos)} documentos")
    return todos_fragmentos


# ── Prueba directa del módulo ────────────────────────────────
# Si ejecutas este archivo directamente, carga data/raw y muestra
# un resumen. Útil para verificar que todo funciona.

if __name__ == "__main__":
    fragmentos = cargar_carpeta("data/raw")

    if fragmentos:
        print("\n" + "=" * 60)
        print("EJEMPLO DE FRAGMENTO (el primero):")
        print("=" * 60)
        primero = fragmentos[0]
        print(f"Fuente:  {primero.metadata['fuente']}")
        print(f"Formato: {primero.metadata['formato']}")
        print(f"Página:  {primero.metadata['pagina']}")
        print(f"Hoja:    {primero.metadata['hoja']}")
        print(f"\nTexto (primeros 300 caracteres):")
        print(primero.texto[:300])
