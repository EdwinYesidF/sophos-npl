"""Ingesta de Sophos NLP: construye el índice FAISS a partir del curso de Hugging Face.

Uso:  python build_index.py
Requiere git. Genera la carpeta definida en config.INDEX_DIR.
"""
import re
import subprocess
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def clean_mdx(text: str) -> str:
    """Limpia un archivo MDX del curso y deja solo texto y código útiles."""
    # Bloques condicionales por framework: se conserva la versión de PyTorch
    text = re.sub(r"\{#if fw === 'pt'\}(.*?)\{:else\}.*?\{/if\}", r"\1", text, flags=re.S)
    text = re.sub(r"\{#if fw === 'tf'\}.*?\{:else\}(.*?)\{/if\}", r"\1", text, flags=re.S)
    text = re.sub(r"\{#if fw === 'tf'\}.*?\{/if\}", "", text, flags=re.S)
    text = re.sub(r"\{#if fw === 'pt'\}(.*?)\{/if\}", r"\1", text, flags=re.S)
    text = re.sub(r"</?[A-Za-z][^>\n]*>", "", text)    # etiquetas HTML/JSX
    text = re.sub(r"\[\[[^\]]+\]\]", "", text)          # anclas de los títulos
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_documents() -> list:
    """Descarga el curso y devuelve un documento limpio por sección."""
    if not Path("course").exists():
        subprocess.run(["git", "clone", "--depth", "1", "-q", config.COURSE_REPO], check=True)
    docs = []
    for n in config.COURSE_CHAPTERS:
        for f in sorted(Path(f"course/chapters/en/chapter{n}").glob("*.mdx")):
            raw = f.read_text(encoding="utf-8")
            if "<Question" in raw:      # se excluyen los cuestionarios de fin de capítulo
                continue
            text = clean_mdx(raw)
            title = re.search(r"^#\s+(.+)$", text, flags=re.M)
            docs.append(Document(page_content=text, metadata={
                "source": f"Hugging Face Course, chapter {n}, section {f.stem}",
                "title": title.group(1).strip() if title else f.stem,
                "url": f"https://huggingface.co/learn/llm-course/chapter{n}/{f.stem}",
            }))
    return docs


def main() -> None:
    docs = load_documents()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    ).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    FAISS.from_documents(chunks, embeddings).save_local(config.INDEX_DIR)
    print(f"Documentos: {len(docs)} | Chunks: {len(chunks)} | Índice: {config.INDEX_DIR}/")


if __name__ == "__main__":
    main()
