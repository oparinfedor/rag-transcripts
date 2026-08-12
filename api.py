"""FastAPI-обёртка над RAG-проектом.

Даёт HTTP-доступ к базе знаний (ask_knowledge_base из rag.py) для внешних
систем — в первую очередь для n8n-агента.

Запуск:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Проверка:
    http://localhost:8000/health
    http://localhost:8000/docs  (автосгенерированная документация)
"""
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from config import (
    ROUTERAI_API_KEY, DATA_DIR, docx_folder, list_projects, get_active_project,
    set_active_project, validate_project,
)
from rag import ask_knowledge_base
from indexing import get_collection, index_all
from parse_docx import parse_all
from summarize import summarize_all

app = FastAPI(title="RAG Transcripts API")


def resolve_project(project: Optional[str]) -> str:
    project = project or get_active_project()
    try:
        validate_project(project)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return project


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    project: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    chunks_in_base: int
    project: str


class SelectProjectRequest(BaseModel):
    project: str


class ClaimUpdateRequest(BaseModel):
    update_id: int


# --- Дедупликация апдейтов Telegram между параллельными выполнениями n8n ---
# Schedule Trigger в n8n опрашивает Telegram раз в 10 сек, а RAG-ответ может
# идти дольше — тогда следующий тик стартует, пока предыдущий ещё не
# закончился, и оба видят один и тот же апдейт как новый (n8n сохраняет
# static data только по завершении выполнения, а не сразу). api.py — один
# процесс с настоящим общим состоянием, поэтому дедуп живёт здесь.
_claim_lock = threading.Lock()
_claimed_update_ids = set()
_claimed_updates_file = DATA_DIR / ".processed_telegram_updates"

if _claimed_updates_file.exists():
    for _line in _claimed_updates_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line:
            _claimed_update_ids.add(int(_line))


@app.post("/telegram/claim-update")
def claim_update(request: ClaimUpdateRequest):
    """Возвращает claimed=true один-единственный раз для каждого update_id.
    Второй (перекрывшийся) прогон воркфлоу получает claimed=false и должен
    молча остановиться, не дублируя RAG-запрос и ответ пользователю."""
    with _claim_lock:
        if request.update_id in _claimed_update_ids:
            return {"claimed": False}

        _claimed_update_ids.add(request.update_id)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_claimed_updates_file, "a", encoding="utf-8") as f:
            f.write(f"{request.update_id}\n")

        if len(_claimed_update_ids) > 2000:
            trimmed = sorted(_claimed_update_ids)[-1000:]
            _claimed_update_ids.clear()
            _claimed_update_ids.update(trimmed)
            _claimed_updates_file.write_text(
                "\n".join(str(i) for i in trimmed) + "\n", encoding="utf-8"
            )

    return {"claimed": True}


@app.get("/health")
def health(project: Optional[str] = None):
    project = resolve_project(project)
    collection = get_collection(project)
    return {"status": "ok", "project": project, "chunks_in_base": collection.count()}


@app.get("/projects")
def projects():
    active = get_active_project()
    result = []
    for name in list_projects():
        result.append({"name": name, "chunks": get_collection(name).count()})
    return {"projects": result, "active": active}


@app.post("/projects/select")
def select_project(request: SelectProjectRequest):
    try:
        created = set_active_project(request.project)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"active": request.project, "created": created}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not ROUTERAI_API_KEY:
        raise HTTPException(status_code=500, detail="Не задан ROUTERAI_API_KEY в .env")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    project = resolve_project(request.project)

    try:
        answer = ask_knowledge_base(request.question, project, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    collection = get_collection(project)
    return QueryResponse(answer=answer, chunks_in_base=collection.count(), project=project)


@app.post("/process/parse")
def process_parse(project: Optional[str] = None):
    """Парсит новые .docx из data/<project>/docx/ в JSON. Идемпотентно —
    уже распарсенные файлы пропускаются. Вызывается из n8n по расписанию
    или после загрузки транскрипта через Telegram-бота. Без параметра
    project работает с активным проектом."""
    return parse_all(resolve_project(project))


@app.post("/process/summarize")
def process_summarize(project: Optional[str] = None):
    if not ROUTERAI_API_KEY:
        raise HTTPException(status_code=500, detail="Не задан ROUTERAI_API_KEY в .env")
    return summarize_all(resolve_project(project))


@app.post("/process/index")
def process_index(project: Optional[str] = None):
    return index_all(resolve_project(project))


@app.post("/upload-transcript")
async def upload_transcript(file: UploadFile = File(...), project: Optional[str] = None):
    """Принимает .docx (например, от Telegram-бота через n8n) и сохраняет
    в data/<project>/docx/, откуда его подхватит /process/parse. Без
    параметра project сохраняет в активный проект."""
    if not file.filename or not file.filename.lower().endswith(".docx"):
        raise HTTPException(status_code=400, detail="Ожидается файл с расширением .docx")

    project = resolve_project(project)
    safe_name = Path(file.filename).name  # защита от path traversal
    folder = docx_folder(project)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / safe_name

    content = await file.read()
    dest.write_bytes(content)

    return {"saved_as": safe_name, "project": project, "size_bytes": len(content)}
