"""Суммаризация транскриптов через RouterAI (Claude Sonnet 5).

Прогоняет все JSON-транскрипты в data/transcripts/, для каждого сохраняет
рядом файл *_summary.json. Уже готовые summary пропускает.
"""
import glob
import json
import time
from pathlib import Path

import requests

from config import ROUTERAI_API_KEY, API_URL, MODEL, SYSTEM_PROMPT, transcripts_folder


def summarize_transcript(transcript_text, api_key=None):
    """api_key: если не передан, берётся ROUTERAI_API_KEY из .env
    (нужно для Streamlit, где ключ может прийти из поля в сайдбаре)."""
    api_key = api_key or ROUTERAI_API_KEY
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "input": f"{SYSTEM_PROMPT}\n\n---\n\nТранскрипт:\n{transcript_text}",
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    if "output_text" in data:
        raw_text = data["output_text"].strip()
    else:
        # запасной вариант: RouterAI не всегда дублирует текст в output_text.
        # output — список блоков; при включённом reasoning первым идёт блок
        # "reasoning" (без поля content), а само сообщение — в блоке "message"
        # на произвольной позиции, поэтому ищем его по type, а не по индексу.
        raw_text = None
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        raw_text = content["text"].strip()
                        break
        if raw_text is None:
            raise RuntimeError(f"Не удалось разобрать ответ RouterAI: {data}")

    # на случай, если модель обернёт ответ в markdown-блок
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]

    return json.loads(raw_text)


def summarize_all(project):
    if not ROUTERAI_API_KEY:
        raise RuntimeError("Не задан ROUTERAI_API_KEY — проверь файл .env")

    all_json = [
        p for p in glob.glob(str(transcripts_folder(project) / "*.json"))
        if not p.endswith("_summary.json")
    ]
    print(f"Найдено транскриптов: {len(all_json)}")

    processed, skipped, failed = 0, 0, 0

    for path in all_json:
        summary_path = Path(path).with_name(Path(path).stem + "_summary.json")
        if summary_path.exists():
            print(f"Пропуск (уже есть summary): {Path(path).name}")
            skipped += 1
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = " ".join(seg["text"] for seg in data["segments"])
            if not text.strip():
                print(f"Пропуск (пустой текст): {Path(path).name}")
                skipped += 1
                continue

            print(f"Обрабатываю: {Path(path).name}...")
            result = summarize_transcript(text)

            summary_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            processed += 1
            time.sleep(1)  # небольшая пауза между запросами

        except Exception as e:
            print(f"  ОШИБКА на файле {Path(path).name}: {e}")
            failed += 1
            continue

    print(f"\nГотово. Обработано: {processed}, пропущено: {skipped}, ошибок: {failed}")
    return {"found": len(all_json), "processed": processed, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    import sys
    from config import get_active_project
    proj = sys.argv[1] if len(sys.argv) > 1 else get_active_project()
    summarize_all(proj)
