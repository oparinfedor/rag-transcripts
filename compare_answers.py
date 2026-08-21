"""
compare_answers.py

Сравнение rag.py (линейная версия) vs graph_rag.py (self-correcting граф)
на одном и том же eval-датасете. В отличие от compare_recall.py (который
сравнивал ChromaDB и Qdrant на уровне retrieval), здесь сравниваются
конечные ОТВЕТЫ — линейная версия делает 1 вызов LLM, графовая — от 2 до
7 в зависимости от того, сработал ли цикл переформулировки.

Что измеряется:
  - латентность (сколько реально заняло по времени)
  - число вызовов LLM (точный прокси на стоимость)
  - source_mentioned: механическая проверка — упомянул ли текст ответа хотя бы
    один из expected_sources по имени (грубая проверка, не замена
    содержательной оценке качества)

Качество ответов по существу скрипт не оценивает — оба ответа печатаются
полностью, чтобы прочитать и сравнить глазами. На 7 вопросах это быстрее
и честнее, чем городить LLM-as-judge поверх LLM-as-judge.

Использование:
    python compare_answers.py --project "<имя_коллекции>"
"""

import argparse
import time
from pathlib import Path

import rag
from config import get_active_project
from eval_embeddings import eval_dataset
from graph_rag import ask_knowledge_base_graph


def mentions_expected_source(answer_text: str, expected_sources: list) -> bool:
    """Грубая проверка: упоминается ли в тексте ответа стем (имя без
    расширения) хотя бы одного ожидаемого источника. Это приближение —
    модель может процитировать содержание без явного упоминания имени
    файла, поэтому False здесь не всегда значит "ответ неверный"."""
    for source in expected_sources:
        stem = Path(source).stem
        if stem in answer_text:
            return True
    return False


def run_linear(question: str, project: str):
    start = time.perf_counter()
    answer = rag.ask_knowledge_base(question, project)
    elapsed = time.perf_counter() - start
    return {"answer": answer, "elapsed": elapsed, "llm_calls": 1, "token_usage": dict(rag.LAST_USAGE)}


def run_graph(question: str, project: str):
    start = time.perf_counter()
    result = ask_knowledge_base_graph(question, project)
    elapsed = time.perf_counter() - start
    result["elapsed"] = elapsed
    return result


def sum_usage(usages: list) -> dict:
    total = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value
    return total


def main():
    parser = argparse.ArgumentParser(description="Сравнение rag.py vs graph_rag.py на eval-датасете")
    parser.add_argument(
        "--project",
        default=None,
        help="Имя коллекции в Chroma. Если не указано — берётся из get_active_project() (т.е. из .env).",
    )
    args = parser.parse_args()
    project = args.project or get_active_project()

    dataset = eval_dataset
    print(f"Загружено {len(dataset)} вопросов из eval_embeddings.py\n")

    linear_times, graph_times = [], []
    linear_calls, graph_calls = [], []
    linear_hits, graph_hits = [], []
    linear_usages, graph_usages = [], []

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item["expected_sources"]

        print("=" * 70)
        print(f"[{i}/{len(dataset)}] Вопрос: {question}")
        print("=" * 70)

        linear = run_linear(question, project)
        graph = run_graph(question, project)

        linear_hit = mentions_expected_source(linear["answer"], expected)
        graph_hit = mentions_expected_source(graph["answer"], expected)

        linear_times.append(linear["elapsed"])
        graph_times.append(graph["elapsed"])
        linear_calls.append(linear["llm_calls"])
        graph_calls.append(graph["llm_calls"])
        linear_hits.append(linear_hit)
        graph_hits.append(graph_hit)
        linear_usages.append(linear["token_usage"])
        graph_usages.append(graph["token_usage"])

        print(f"\n--- ЛИНЕЙНАЯ (rag.py) | {linear['elapsed']:.1f}s | {linear['llm_calls']} вызов(ов) LLM | "
              f"usage={linear['token_usage']} | source_mentioned={linear_hit} ---")
        print(linear["answer"])

        print(f"\n--- ГРАФ (graph_rag.py) | {graph['elapsed']:.1f}s | {graph['llm_calls']} вызов(ов) LLM | "
              f"retries={graph['retry_count']} | usage={graph['token_usage']} | source_mentioned={graph_hit} ---")
        print(graph["answer"])
        print()

    print("\n" + "=" * 70)
    print("ИТОГО")
    print("=" * 70)
    print(f"Линейная (rag.py):    среднее время {sum(linear_times)/len(linear_times):.1f}s, "
          f"среднее число вызовов LLM {sum(linear_calls)/len(linear_calls):.1f}, "
          f"source_mentioned в {sum(linear_hits)}/{len(linear_hits)} ответах, "
          f"суммарный usage: {sum_usage(linear_usages)}")
    print(f"Граф (graph_rag.py):  среднее время {sum(graph_times)/len(graph_times):.1f}s, "
          f"среднее число вызовов LLM {sum(graph_calls)/len(graph_calls):.1f}, "
          f"source_mentioned в {sum(graph_hits)}/{len(graph_hits)} ответах, "
          f"суммарный usage: {sum_usage(graph_usages)}")

    linear_total = sum_usage(linear_usages)
    graph_total = sum_usage(graph_usages)
    if not linear_total or not graph_total:
        print("\nПРИМЕЧАНИЕ: usage пустой у одной или обеих версий — либо патч в rag.py "
              "не применён, либо RouterAI не возвращает поле usage в этом формате ответа. "
              "Число вызовов LLM (llm_calls) выше остаётся рабочим прокси на стоимость.")


if __name__ == "__main__":
    main()
