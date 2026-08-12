"""Парсинг .docx-транскриптов в JSON с репликами по спикерам.

Формат реплики в файлах: 'Имя: текст' в начале абзаца.
Положи исходные .docx в папку data/docx/, результат появится в data/transcripts/.
"""
import re
import json
import glob
from pathlib import Path
from docx import Document

from config import docx_folder, transcripts_folder

SPEAKER_PATTERN = re.compile(r'^([^:]{1,40}):\s*(.*)$')


def parse_docx(path):
    doc = Document(path)
    segments = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        match = SPEAKER_PATTERN.match(text)
        if match:
            speaker = match.group(1).strip()
            reply_text = match.group(2).strip()
            segments.append({"speaker": speaker, "text": reply_text})
        else:
            # продолжение реплики того же спикера (перенос строки)
            if segments:
                segments[-1]["text"] += " " + text
            else:
                segments.append({"speaker": None, "text": text})

    return segments


def parse_all(project):
    """Парсит все .docx из data/<project>/docx/, кроме уже распарсенных
    (проверка по mtime: если рядом уже есть .json новее исходного .docx —
    пропускаем). Вызывается по расписанию из n8n, поэтому должна быть
    дешёвой при повторном запуске без новых файлов."""
    out_folder = transcripts_folder(project)
    out_folder.mkdir(parents=True, exist_ok=True)
    docx_files = glob.glob(str(docx_folder(project) / "*.docx"))
    print(f"Найдено файлов: {len(docx_files)}")

    parsed, skipped = 0, 0

    for docx_path in docx_files:
        out_path = out_folder / (Path(docx_path).stem + ".json")
        if out_path.exists() and out_path.stat().st_mtime >= Path(docx_path).stat().st_mtime:
            skipped += 1
            continue

        segments = parse_docx(docx_path)
        result = {"source_file": docx_path, "segments": segments}

        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  {Path(docx_path).name} -> {len(segments)} реплик -> {out_path.name}")
        parsed += 1

    print(f"\nГотово. Распарсено: {parsed}, пропущено (уже актуально): {skipped}.")
    return {"found": len(docx_files), "parsed": parsed, "skipped": skipped}


if __name__ == "__main__":
    import sys
    from config import get_active_project
    proj = sys.argv[1] if len(sys.argv) > 1 else get_active_project()
    parse_all(proj)
