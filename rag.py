"""RAG-запрос: находим релевантные чанки в ChromaDB, просим Claude
(через RouterAI) ответить на их основе с указанием источника.
"""
from pathlib import Path

import requests

from config import ROUTERAI_API_KEY, API_URL, MODEL
from indexing import get_embed_model, get_collection

LAST_USAGE = {}

def ask_knowledge_base(question, project, top_k=5, api_key=None):
    """api_key: если не передан, берётся ROUTERAI_API_KEY из .env
    (нужно для Streamlit, где ключ может прийти из поля в сайдбаре)."""
    api_key = api_key or ROUTERAI_API_KEY
    embed_model = get_embed_model()
    collection = get_collection(project)

    query_embedding = embed_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    context_blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source_name = Path(meta["source_file"]).stem
        context_blocks.append(f"[Источник: {source_name}]\n{doc}")

    context = "\n\n---\n\n".join(context_blocks)

    rag_prompt = f"""Отвечай на вопрос ТОЛЬКО на основе приведённого ниже контекста.
Если ответа в контексте нет - честно скажи, что не нашёл информации, не выдумывай.
В конце ответа обязательно укажи, из каких источников (названия файлов) взята информация.

Контекст:
{context}

Вопрос: {question}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": rag_prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    global LAST_USAGE
    LAST_USAGE = data.get("usage", {})

    if "output_text" in data:
        return data["output_text"]

    # запасной вариант: RouterAI не всегда дублирует текст в output_text.
    # output — список блоков; при включённом reasoning первым идёт блок
    # "reasoning" (без поля content), а само сообщение — в блоке "message"
    # на произвольной позиции, поэтому ищем его по type, а не по индексу.
    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content["text"]

    raise RuntimeError(f"Не удалось разобрать ответ RouterAI: {data}")


if __name__ == "__main__":
    import sys
    from config import get_active_project
    proj = sys.argv[1] if len(sys.argv) > 1 else get_active_project()
    q = input("Вопрос: ")
    print(ask_knowledge_base(q, proj))
