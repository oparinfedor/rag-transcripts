"""Расшифровка аудио/видео через faster-whisper.

ОПЦИОНАЛЬНЫЙ модуль — не входит в requirements.txt по умолчанию,
так как faster-whisper тянет за собой torch (тяжёлая библиотека).
Нужен только если появятся НОВЫЕ записи для распознавания, а не для
уже готовых текстовых транскриптов.

Установка при необходимости: pip install faster-whisper torch

Использование:
    python transcribe.py путь/к/файлу.mp4 [project]
"""
import json
from pathlib import Path

from config import get_active_project, transcripts_folder


def transcribe_audio(audio_path, project=None, language="ru"):
    project = project or get_active_project()
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model_size = "medium" if device == "cuda" else "small"

    print(f"Устройство: {device}, модель: {model_size}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, info = model.transcribe(audio_path, language=language, vad_filter=True)

    result = {
        "audio_file": str(audio_path),
        "language": info.language,
        "duration_seconds": round(info.duration, 1),
        "segments": [],
    }

    for seg in segments:
        result["segments"].append(
            {"start": round(seg.start, 1), "end": round(seg.end, 1), "text": seg.text.strip()}
        )
        print(f"[{seg.start:7.1f}s -> {seg.end:7.1f}s] {seg.text.strip()}")

    folder = transcripts_folder(project)
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / (Path(audio_path).stem + ".json")
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово: {out_path}")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Использование: python transcribe.py путь_к_аудио.mp4 [project]")
    else:
        project_arg = sys.argv[2] if len(sys.argv) > 2 else None
        transcribe_audio(sys.argv[1], project=project_arg)
