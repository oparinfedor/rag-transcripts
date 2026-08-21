"""
compare_recall.py

Сравнение ChromaDB vs Qdrant на одном и том же eval-датасете (тот же формат,
что в eval_embeddings.py — вопросы с expected_sources). Эмбеддинги и векторы
в обеих базах идентичны (Qdrant заполнен через qdrant_indexing.py без
пересчёта), поэтому recall@k должен совпадать — интересны расхождения
в скорости и удобстве работы с метаданными.

Использование:
    python compare_recall.py \
        --project "<имя_коллекции_в_chroma>" \
        --qdrant-collection transcripts_qdrant \
        --metadata-key source

Датасет берётся напрямую из eval_embeddings.py (переменная eval_dataset),
без промежуточного JSON-файла — так меньше шансов рассинхронизировать
данные с тем, что уже используется в основном eval-скрипте.

ВАЖНО: строка с get_embed_model() ниже предполагает, что она возвращает
SentenceTransformer-совместимый объект с методом .encode(text). Если в
indexing.py интерфейс другой — поправь только функцию embed_query().
"""

import argparse
import ntpath
import time

from indexing import get_embed_model, get_collection
from config import get_active_project
from eval_embeddings import eval_dataset
from qdrant_client import QdrantClient


def normalize_source(value: str) -> str:
    """metadata хранит полный Windows-путь (source_file), а expected_sources
    в eval-датасете — голые имена файлов. Приводим к общему виду через basename.
    ntpath используется вместо os.path, чтобы Windows-пути парсились корректно
    независимо от ОС, на которой запускается сам скрипт."""
    if value is None:
        return value
    return ntpath.basename(value)


def embed_query(model, text: str):
    # Поправь эту строку, если get_embed_model() возвращает не
    # SentenceTransformer, а что-то с другим интерфейсом.
    return model.encode(text).tolist()


def hit_rate(retrieved_sources: set, expected_sources: list) -> float:
    """Было ли найдено хотя бы ОДНО совпадение с ожидаемыми источниками в топ-k.
    Это метрика "да/нет" (1.0 или 0.0), а не доля от всех expected_sources —
    именно она стоит за прежними цифрами в README (recall@5 = 100%), когда
    expected_sources был широким списком (20-30 файлов) и строгая формула
    hits/len(expected_sources) физически не могла давать высокие значения
    при top_k=5."""
    if not expected_sources:
        return None
    hits = retrieved_sources & set(expected_sources)
    return 1.0 if hits else 0.0


def query_chroma(collection, embedding, top_k: int, metadata_key: str):
    start = time.perf_counter()
    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["metadatas"],
    )
    elapsed = time.perf_counter() - start
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    sources = {normalize_source(m.get(metadata_key)) for m in metadatas if m.get(metadata_key)}
    return sources, elapsed


def query_qdrant(client: QdrantClient, collection_name: str, embedding, top_k: int, metadata_key: str):
    start = time.perf_counter()
    response = client.query_points(
        collection_name=collection_name,
        query=embedding,
        limit=top_k,
    )
    elapsed = time.perf_counter() - start
    result = response.points
    sources = {normalize_source(point.payload.get(metadata_key)) for point in result if point.payload.get(metadata_key)}
    return sources, elapsed


def main():
    parser = argparse.ArgumentParser(description="Сравнение recall@k и скорости: Chroma vs Qdrant")
    parser.add_argument(
        "--project",
        default=None,
        help="Имя коллекции в Chroma. Если не указано — берётся из get_active_project() (т.е. из .env).",
    )
    parser.add_argument("--qdrant-collection", default="transcripts_qdrant")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--metadata-key", default="source", help="Поле в metadata с именем источника/файла")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--grpc",
        action="store_true",
        help="Использовать gRPC вместо HTTP для запросов к Qdrant — меньше сетевых "
             "накладных расходов, честнее для сравнения скорости самого поиска.",
    )
    args = parser.parse_args()

    dataset = eval_dataset
    print(f"Загружено {len(dataset)} вопросов из eval_embeddings.py")

    embed_model = get_embed_model()
    project = args.project or get_active_project()
    chroma_collection = get_collection(project)
    qdrant_client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, prefer_grpc=args.grpc)
    if args.grpc:
        print("Использую gRPC-транспорт для запросов к Qdrant.")

    # Прогревочный запрос: первый вызов обычно медленнее (устанавливается
    # соединение/прогревается модель), и это не имеет отношения к разнице
    # между базами — не учитываем его в статистике.
    warmup_embedding = embed_query(embed_model, dataset[0]["question"])
    query_chroma(chroma_collection, warmup_embedding, args.top_k, args.metadata_key)
    query_qdrant(qdrant_client, args.qdrant_collection, warmup_embedding, args.top_k, args.metadata_key)
    print("Прогрев выполнен, начинаю замеры...\n")

    chroma_recalls, qdrant_recalls = [], []
    chroma_times, qdrant_times = [], []

    for item in dataset:
        question = item["question"]
        expected = item["expected_sources"]
        embedding = embed_query(embed_model, question)

        chroma_sources, chroma_t = query_chroma(chroma_collection, embedding, args.top_k, args.metadata_key)
        qdrant_sources, qdrant_t = query_qdrant(
            qdrant_client, args.qdrant_collection, embedding, args.top_k, args.metadata_key
        )

        chroma_r = hit_rate(chroma_sources, expected)
        qdrant_r = hit_rate(qdrant_sources, expected)

        if chroma_r is not None:
            chroma_recalls.append(chroma_r)
        if qdrant_r is not None:
            qdrant_recalls.append(qdrant_r)
        chroma_times.append(chroma_t)
        qdrant_times.append(qdrant_t)

        print(f"\nВопрос: {question[:60]}...")
        print(f"  Chroma hit@{args.top_k}: {chroma_r:.2f}  ({chroma_t*1000:.1f} ms)")
        print(f"  Qdrant hit@{args.top_k}: {qdrant_r:.2f}  ({qdrant_t*1000:.1f} ms)")
        if chroma_sources != qdrant_sources:
            print(f"  ВНИМАНИЕ: разные найденные источники — стоит перепроверить перенос данных.")
            print(f"    Chroma: {chroma_sources}")
            print(f"    Qdrant: {qdrant_sources}")

    print("\n" + "=" * 50)
    print("ИТОГО")
    print("=" * 50)
    print(f"Chroma: hit_rate@{args.top_k} = {sum(chroma_recalls)/len(chroma_recalls):.2%}, "
          f"средняя латентность = {sum(chroma_times)/len(chroma_times)*1000:.1f} ms")
    print(f"Qdrant: hit_rate@{args.top_k} = {sum(qdrant_recalls)/len(qdrant_recalls):.2%}, "
          f"средняя латентность = {sum(qdrant_times)/len(qdrant_times)*1000:.1f} ms")


if __name__ == "__main__":
    main()
