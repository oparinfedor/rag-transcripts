"""
qdrant_indexing.py

Параллельная индексация в Qdrant поверх уже посчитанных эмбеддингов из ChromaDB.
Векторы НЕ пересчитываются — переносятся 1:1 из существующей коллекции,
чтобы сравнение Chroma vs Qdrant было честным (одинаковые эмбеддинги с обеих сторон,
разница будет только в самой базе).

Использование:
    python qdrant_indexing.py \
        --chroma-path ./chroma_db \
        --chroma-collection <имя_твоей_коллекции_из_config.py> \
        --qdrant-collection transcripts_qdrant

Требования:
    pip install qdrant-client chromadb --break-system-packages
"""

import argparse
import uuid

import chromadb
from config import get_active_project
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def get_chroma_data(chroma_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)

    data = collection.get(include=["embeddings", "documents", "metadatas"])
    ids = data["ids"]
    embeddings = data["embeddings"]
    documents = data["documents"]
    metadatas = data["metadatas"]

    if embeddings is None or len(embeddings) == 0:
        raise ValueError(
            f"Коллекция '{collection_name}' пуста или не содержит эмбеддингов. "
            "Проверь --chroma-path и точное имя коллекции (см. config.py)."
        )

    return ids, embeddings, documents, metadatas


def ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Коллекция '{collection_name}' уже есть в Qdrant — пересоздаю с нуля.")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def chroma_id_to_uuid(chroma_id: str) -> str:
    """
    Qdrant требует id точки быть unsigned int или UUID (в отличие от Chroma,
    где id — произвольная строка). Детерминированно превращаем chroma_id в UUID5:
    при повторном запуске одна и та же запись перезапишется, а не задублируется.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chroma_id))


def migrate(
    chroma_path: str,
    chroma_collection: str,
    qdrant_host: str,
    qdrant_port: int,
    qdrant_collection: str,
    batch_size: int = 256,
):
    print(f"Читаю данные из ChromaDB: {chroma_path} / коллекция '{chroma_collection}'")
    ids, embeddings, documents, metadatas = get_chroma_data(chroma_path, chroma_collection)
    vector_size = len(embeddings[0])
    print(f"Найдено {len(ids)} точек, размерность вектора: {vector_size}")

    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
    ensure_qdrant_collection(qdrant, qdrant_collection, vector_size)

    points = []
    for chroma_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
        payload = dict(metadata or {})
        payload["document"] = document
        payload["chroma_id"] = chroma_id  # сохраняем исходный id для сверки при отладке

        points.append(
            PointStruct(
                id=chroma_id_to_uuid(chroma_id),
                vector=embedding,
                payload=payload,
            )
        )

    print(f"Загружаю {len(points)} точек в Qdrant батчами по {batch_size}...")
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant.upsert(collection_name=qdrant_collection, points=batch)
        print(f"  загружено {min(i + batch_size, len(points))}/{len(points)}")

    count = qdrant.count(qdrant_collection).count
    print(f"Готово. В коллекции '{qdrant_collection}' сейчас {count} точек.")

    if count != len(ids):
        print(
            f"ВНИМАНИЕ: количество точек в Qdrant ({count}) не совпадает "
            f"с количеством в Chroma ({len(ids)}) — стоит перепроверить перед сравнением recall@5."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Перенос эмбеддингов из ChromaDB в Qdrant без пересчёта векторов."
    )
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help="Имя коллекции в Chroma. Если не указано — берётся из get_active_project() (т.е. из .env).",
    )
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--qdrant-collection", default="transcripts_qdrant")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    chroma_collection = args.chroma_collection or get_active_project()

    migrate(
        chroma_path=args.chroma_path,
        chroma_collection=chroma_collection,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        qdrant_collection=args.qdrant_collection,
        batch_size=args.batch_size,
    )
