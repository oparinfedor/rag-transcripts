"""Общая конфигурация проекта.

Заменяет Colab-переменные (пути на Google Drive, getpass для ключа)
на локальные пути внутри проекта и .env-файл для ключа.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# --- Персистентная векторная БД ---
# Хранится локально на диске, НЕ пересоздаётся при каждом запуске
# (в отличие от Colab-сессии, которая обнулялась). Общая на все проекты —
# изоляция между проектами достигается отдельной ChromaDB-коллекцией
# на каждый, а не отдельной папкой.
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# --- Проекты (изолированные клиентские базы) ---
# Проект = слаг, он же имя папки под data/<project>/ и имя коллекции
# ChromaDB. Список проектов нигде не хранится отдельно — это просто
# папки под data/, у которых есть docx/.
PROJECT_SLUG_RE = re.compile(r"^[a-zA-Z0-9_-]{1,50}$")
ACTIVE_PROJECT_FILE = DATA_DIR / ".active_project"
DEFAULT_PROJECT = os.environ.get("DEFAULT_PROJECT", "demo")


def validate_project(name):
    if not name or not PROJECT_SLUG_RE.match(name):
        raise ValueError(
            f"Некорректное имя проекта: {name!r}. "
            "Разрешены только буквы, цифры, - и _ (до 50 символов)."
        )
    return name


def docx_folder(project):
    validate_project(project)
    return DATA_DIR / project / "docx"


def transcripts_folder(project):
    validate_project(project)
    return DATA_DIR / project / "transcripts"


def list_projects():
    if not DATA_DIR.exists():
        return []
    return sorted(
        p.name for p in DATA_DIR.iterdir()
        if p.is_dir() and (p / "docx").exists()
    )


def get_active_project():
    if ACTIVE_PROJECT_FILE.exists():
        return ACTIVE_PROJECT_FILE.read_text(encoding="utf-8").strip()
    return DEFAULT_PROJECT


def set_active_project(project):
    validate_project(project)
    created = not docx_folder(project).exists()
    docx_folder(project).mkdir(parents=True, exist_ok=True)
    transcripts_folder(project).mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_PROJECT_FILE.write_text(project, encoding="utf-8")
    return created

# --- RouterAI ---
ROUTERAI_API_KEY = os.environ.get("ROUTERAI_API_KEY")
API_URL = "https://routerai.ru/api/v1/responses"
MODEL = "anthropic/claude-sonnet-5"

# --- Чанкинг для RAG ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Модель эмбеддингов ---
EMBEDDING_MODEL = "BAAI/bge-m3"

SYSTEM_PROMPT = """Ты помогаешь анализировать транскрипты рабочих встреч.
Тебе дают текст расшифровки встречи (может быть без разметки по спикерам).
Верни ТОЛЬКО валидный JSON без каких-либо пояснений и markdown-разметки,
строго в следующей структуре:

{
  "summary": "краткое содержание встречи в 3-5 предложениях",
  "themes": ["тема 1", "тема 2", "тема 3"],
  "key_points": ["ключевой момент 1", "ключевой момент 2"],
  "quotes": ["важная цитата 1 из текста", "важная цитата 2 из текста"]
}

Если в тексте есть явные ошибки распознавания речи (случайные слова,
не подходящие по смыслу) - не включай их в summary дословно,
восстанавливай смысл по контексту."""
