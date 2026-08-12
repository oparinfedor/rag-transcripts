"""Чанкинг + embeddings + персистентная векторная БД (ChromaDB).

Главное отличие от Colab-версии: база хранится в локальной папке
chroma_db/ внутри проекта и НЕ пересоздаётся заново при каждом запуске —
именно то, что решает исходную проблему непостоянного хранилища.
"""
import re
import glob
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, transcripts_folder,
    validate_project,
)

_embed_model = None
_chroma_client = None
_collections = {}


def get_embed_model():
    """Ленивая загрузка модели эмбеддингов (один раз на процесс, общая
    для всех проектов — модель project-agnostic)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def get_collection(project):
    """Подключение к персистентной коллекции ChromaDB конкретного проекта.
    Один CHROMA_PATH на диске, но у каждого проекта своя именованная
    коллекция — Chroma изолирует их полностью."""
    global _chroma_client
    validate_project(project)
    if project not in _collections:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collections[project] = _chroma_client.get_or_create_collection(project)
    return _collections[project]


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Режем по предложениям, набирая чанк примерно до chunk_size символов,
    с небольшим перехлёстом (overlap), чтобы не рвать мысль на границе."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += (" " if current else "") + sent
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + " " + sent

    if current:
        chunks.append(current.strip())

    return chunks


def index_all(project):
    """Индексирует все транскрипты из data/<project>/transcripts/, кроме
    уже проиндексированных (проверка по source_file в метаданных этой же
    коллекции — так что дубликат имени файла в другом проекте не мешает)."""
    embed_model = get_embed_model()
    collection = get_collection(project)

    all_json_files = [
        p for p in glob.glob(str(transcripts_folder(project) / "*.json"))
        if not p.endswith("_summary.json")
    ]

    already_indexed = set()
    existing = collection.get(include=["metadatas"])
    for meta in existing["metadatas"]:
        already_indexed.add(meta["source_file"])

    added_files, skipped_files = 0, 0

    for path in all_json_files:
        if path in already_indexed:
            skipped_files += 1
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_text = " ".join(seg["text"] for seg in data["segments"])
        if not full_text.strip():
            continue

        chunks = split_into_chunks(full_text)
        embeddings = embed_model.encode(chunks).tolist()

        ids = [f"{Path(path).stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source_file": path, "chunk_index": i} for i in range(len(chunks))]

        collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)

        added_files += 1
        print(f"Проиндексирован: {Path(path).name} ({len(chunks)} чанков)")

    print(f"\nГотово. Новых файлов: {added_files}, пропущено (уже в базе): {skipped_files}")
    print(f"Всего чанков в базе: {collection.count()}")
    return {
        "new_files": added_files,
        "skipped_files": skipped_files,
        "total_chunks": collection.count(),
    }


if __name__ == "__main__":
    import sys
    from config import get_active_project
    proj = sys.argv[1] if len(sys.argv) > 1 else get_active_project()
    index_all(proj)
