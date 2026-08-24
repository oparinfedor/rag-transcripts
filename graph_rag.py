"""
graph_rag.py

Параллельная версия ask_knowledge_base() из rag.py — та же логика
(эмбеддинг вопроса, поиск в Chroma, вызов LLM через RouterAI), но
разложенная на узлы LangGraph с двумя механиками, которых нет в линейной
версии: самопроверка достаточности контекста с циклом, и персистентная
память между отдельными вызовами (сессиями).

После поиска отдельный узел (grade) честно оценивает, отвечает ли
найденный контекст на вопрос. Если нет — запрос переформулируется
(rewrite) и поиск повторяется (с ограничением на число попыток, чтобы
не зациклиться). Каждая повторная попытка исключает уже виденные чанки
и накапливает контекст — так grade/generate в итоге видят объединение
всех уникальных чанков, найденных за все попытки, а не только последние.

Память реализована через штатный механизм LangGraph — checkpointer
(SqliteSaver), а не самодельным способом. История вопросов/ответов
пишется в conversation_memory.db и переживает перезапуск процесса:
два отдельных запуска скрипта с одним и тем же thread_id продолжают
один и тот же диалог, а с разными thread_id — независимые сессии,
не видящие историю друг друга.

(Раньше здесь был ещё узел router, решавший, нужен ли вообще поиск —
убран после того, как сравнение на eval-датасете показало, что он на
одном из 7 вопросов ошибочно увёл предметный вопрос мимо базы. Модель
роутинга и её ограничения — тема отдельного черновика решения в
README, не самого кода.)

Граф:
    START -> retrieve -> grade -> generate -> END
                 ^                  |
                 |                  v (если grade=INSUFFICIENT и retries < max)
                 +------ rewrite <--+

Использование:
    python graph_rag.py <project> [thread_id]

Требования:
    pip install langgraph langgraph-checkpoint-sqlite --break-system-packages
"""

import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict
import operator

import requests

from config import API_URL, MODEL, ROUTERAI_API_KEY
from indexing import get_collection, get_embed_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class RAGState(TypedDict):
    original_question: str
    current_query: str
    project: str
    top_k: int
    api_key: Optional[str]
    context_blocks: List[str]
    sources: List[str]
    seen_chunk_ids: List[str]
    grade: Optional[str]
    retry_count: int
    max_retries: int
    answer: Optional[str]
    token_usage: dict
    # Annotated с operator.add — LangGraph при чекпоинтинге ДОБАВЛЯЕТ то,
    # что возвращает узел, к уже сохранённому списку, а не перезаписывает
    # его. Именно это и даёт память между отдельными invoke() на одном
    # thread_id: каждый новый вопрос дописывает свою пару в общую историю,
    # не стирая предыдущие.
    conversation_history: Annotated[List[str], operator.add]


def call_llm(prompt: str, api_key: Optional[str] = None) -> tuple[str, dict]:
    """Вынесено из ask_knowledge_base() в rag.py — та же логика вызова
    RouterAI и разбора ответа, переиспользуется всеми узлами графа.
    Возвращает (текст_ответа, usage) — usage берётся из data["usage"],
    если API его возвращает (структура зависит от того, как RouterAI
    проксирует ответ; если поля нет — возвращается пустой dict, и это
    не ошибка, просто для этого вызова стоимость в токенах неизвестна)."""
    api_key = api_key or ROUTERAI_API_KEY
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": MODEL, "input": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    usage = data.get("usage", {})

    if "output_text" in data:
        return data["output_text"], usage

    for item in data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return content["text"], usage

    raise RuntimeError(f"Не удалось разобрать ответ RouterAI: {data}")


def merge_usage(accumulated: dict, new: dict) -> dict:
    """Суммирует числовые поля usage между вызовами (input_tokens,
    output_tokens, total_tokens — или как они там называются в ответе
    RouterAI, специально не захардкожено под конкретные имена ключей,
    чтобы не сломаться, если названия полей отличаются)."""
    merged = dict(accumulated)
    for key, value in (new or {}).items():
        if isinstance(value, (int, float)):
            merged[key] = merged.get(key, 0) + value
    return merged


# --- Узлы графа ---

def retrieve_node(state: RAGState) -> dict:
    embed_model = get_embed_model()
    collection = get_collection(state["project"])

    seen_ids = set(state.get("seen_chunk_ids", []))
    # Перезапрашиваем с запасом на число уже виденных чанков — чтобы после
    # фильтрации всё равно осталось top_k новых, не пересекающихся с
    # предыдущими попытками.
    fetch_n = state["top_k"] + len(seen_ids)

    query_embedding = embed_model.encode([state["current_query"]]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=fetch_n)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    candidates = [
        (chunk_id, doc, meta)
        for chunk_id, doc, meta in zip(ids, docs, metas)
        if chunk_id not in seen_ids
    ]
    new_chunks = candidates[: state["top_k"]]

    if not new_chunks:
        # Все найденные чанки уже видели на прошлых попытках (база
        # маленькая или запросы семантически слишком похожи) — берём
        # исходный топ без фильтрации, чтобы не остаться совсем без
        # контекста.
        new_chunks = list(zip(ids, docs, metas))[: state["top_k"]]
        print("[retrieve] все найденные чанки уже виделись ранее — фильтрация отключена для этой попытки")

    context_blocks = list(state.get("context_blocks", []))
    sources = list(state.get("sources", []))
    new_ids = []
    for chunk_id, doc, meta in new_chunks:
        source_name = Path(meta["source_file"]).stem
        sources.append(source_name)
        new_ids.append(chunk_id)
        context_blocks.append(f"[Источник: {source_name}]\n{doc}")

    print(f"[retrieve] найдено {len(new_chunks)} новых чанков по запросу: {state['current_query'][:60]} "
          f"(всего накоплено чанков: {len(context_blocks)})")

    return {
        "context_blocks": context_blocks,
        "sources": sources,
        "seen_chunk_ids": list(seen_ids | set(new_ids)),
    }


def grade_node(state: RAGState) -> dict:
    context = "\n\n---\n\n".join(state["context_blocks"])
    prompt = f"""Отвечает ли приведённый ниже контекст на вопрос пользователя? Оцени честно.
Отвечай ТОЛЬКО одним словом: SUFFICIENT или INSUFFICIENT.

Вопрос: {state['original_question']}

Контекст:
{context}

Ответ (одно слово):"""
    raw, usage = call_llm(prompt, state.get("api_key"))
    raw = raw.strip().upper()
    # INSUFFICIENT содержит SUFFICIENT как подстроку — проверяем длинный вариант первым
    grade = "INSUFFICIENT" if "INSUFFICIENT" in raw else "SUFFICIENT"
    print(f"[grade] оценка: {grade} (попытка {state['retry_count'] + 1})")
    return {"grade": grade, "token_usage": merge_usage(state.get("token_usage", {}), usage)}


def rewrite_node(state: RAGState) -> dict:
    prompt = f"""Найденный контекст не раскрыл вопрос пользователя. Переформулируй вопрос
в более эффективный поисковый запрос (другие слова/синонимы/более общая формулировка).
Ответь ТОЛЬКО текстом нового запроса, без пояснений.

Исходный вопрос: {state['original_question']}
Предыдущий поисковый запрос: {state['current_query']}

Новый поисковый запрос:"""
    new_query, usage = call_llm(prompt, state.get("api_key"))
    new_query = new_query.strip()
    print(f"[rewrite] новый запрос: {new_query[:60]}")
    return {
        "current_query": new_query,
        "retry_count": state["retry_count"] + 1,
        "token_usage": merge_usage(state.get("token_usage", {}), usage),
    }


def generate_node(state: RAGState) -> dict:
    context = "\n\n---\n\n".join(state["context_blocks"])
    history = state.get("conversation_history", [])
    history_block = ""
    if history:
        history_block = (
            "История предыдущих вопросов и ответов в этой сессии (используй, "
            "если текущий вопрос ссылается на них — например, «а по нему», "
            "«сравни с предыдущим»):\n\n" + "\n\n".join(history) + "\n\n---\n\n"
        )

    prompt = f"""Отвечай на вопрос ТОЛЬКО на основе приведённого ниже контекста.
Если ответа в контексте нет - честно скажи, что не нашёл информации, не выдумывай.
В конце ответа обязательно укажи, из каких источников (названия файлов) взята информация.

{history_block}Контекст:
{context}

Вопрос: {state['original_question']}"""

    answer, usage = call_llm(prompt, state.get("api_key"))
    print("[generate] ответ сформирован")
    return {
        "answer": answer,
        "token_usage": merge_usage(state.get("token_usage", {}), usage),
        "conversation_history": [f"Вопрос: {state['original_question']}\nОтвет: {answer}"],
    }


# --- Условные переходы ---

def route_after_grade(state: RAGState) -> str:
    if state["grade"] == "SUFFICIENT" or state["retry_count"] >= state["max_retries"]:
        return "generate"
    return "rewrite"


def build_graph(checkpointer=None):
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)


_checkpoint_conn = None


def _get_checkpointer():
    """Открывает SQLite-соединение для памяти один раз на процесс и
    переиспользует его между вызовами — так conversation_memory.db не
    открывается заново на каждый вопрос."""
    global _checkpoint_conn
    if _checkpoint_conn is None:
        _checkpoint_conn = sqlite3.connect("conversation_memory.db", check_same_thread=False)
    return SqliteSaver(_checkpoint_conn)


def ask_knowledge_base_graph(
    question: str,
    project: str,
    top_k: int = 5,
    api_key: str = None,
    max_retries: int = 2,
    thread_id: str = None,
) -> dict:
    """Возвращает словарь с ответом и метаданными выполнения (нужно для
    сравнения со стоимостью линейной версии — см. compare_answers.py):
      - answer: текст ответа
      - retry_count: сколько раз сработал цикл rewrite -> retrieve
      - sources: источники, накопленные за все попытки retrieve (без дублей)
      - llm_calls: сколько раз всего был вызван LLM (grade/rewrite/generate)
      - token_usage: суммарный usage за этот вызов
      - thread_id: id сессии — передай его же в следующий вызов, чтобы
        продолжить тот же диалог (память подтянется из conversation_memory.db,
        переживает перезапуск процесса). Если не передать — каждый вызов
        независим, как было раньше (используется compare_answers.py).
    """
    thread_id = thread_id or str(uuid.uuid4())
    checkpointer = _get_checkpointer()
    app = build_graph(checkpointer)

    initial_state: RAGState = {
        "original_question": question,
        "current_query": question,
        "project": project,
        "top_k": top_k,
        "api_key": api_key,
        "context_blocks": [],
        "sources": [],
        "seen_chunk_ids": [],
        "grade": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "answer": None,
        "token_usage": {},
        "conversation_history": [],
    }
    config = {"configurable": {"thread_id": thread_id}}
    final_state = app.invoke(initial_state, config)

    retry_count = final_state["retry_count"]
    # grade на каждую попытку (retry_count+1) + rewrite на каждый повтор (retry_count) + generate(1)
    llm_calls = (retry_count + 1) + retry_count + 1

    return {
        "answer": final_state["answer"],
        "retry_count": retry_count,
        "sources": final_state["sources"],
        "llm_calls": llm_calls,
        "token_usage": final_state["token_usage"],
        "thread_id": thread_id,
    }


if __name__ == "__main__":
    from config import get_active_project

    proj = sys.argv[1] if len(sys.argv) > 1 else get_active_project()
    thread_id = sys.argv[2] if len(sys.argv) > 2 else None

    q = input("Вопрос: ")
    result = ask_knowledge_base_graph(q, proj, thread_id=thread_id)
    print(f"\n{result['answer']}")
    print(f"\n[метаданные] retry_count={result['retry_count']}, llm_calls={result['llm_calls']}, "
          f"token_usage={result['token_usage']}")
    print(f"[сессия] thread_id={result['thread_id']} — передай его вторым аргументом "
          f"следующему запуску, чтобы продолжить этот же диалог")