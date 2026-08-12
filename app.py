"""Streamlit-интерфейс: загрузка .docx транскриптов, суммаризация,
добавление в RAG-базу и поиск по ней.

Запуск локально (без ngrok/туннелей — не нужны, приложение уже на компьютере):
    streamlit run app.py

Использует уже готовые модули проекта вместо дублирования логики:
parse_docx.py, summarize.py, indexing.py, rag.py, config.py.
"""
import os
import json

import streamlit as st

from config import ROUTERAI_API_KEY, list_projects, get_active_project, validate_project
from parse_docx import parse_docx
from summarize import summarize_transcript
from indexing import get_embed_model, get_collection, split_into_chunks
from rag import ask_knowledge_base


def get_api_key():
    """Сначала пробуем .env, иначе просим в сайдбаре."""
    key = ROUTERAI_API_KEY or os.environ.get("ROUTERAI_API_KEY")
    if not key:
        key = st.sidebar.text_input("RouterAI API-ключ", type="password")
    return key


@st.cache_resource
def load_embed_model():
    return get_embed_model()


@st.cache_resource
def load_chroma_collection(project):
    return get_collection(project)


def add_to_rag(source_name, full_text, collection, embed_model, progress_callback=None):
    chunks = split_into_chunks(full_text)

    # считаем embeddings небольшими партиями, а не всё разом -
    # это даёт браузеру регулярные обновления прогресса
    BATCH = 10
    all_embeddings = []
    for i in range(0, len(chunks), BATCH):
        batch_chunks = chunks[i:i + BATCH]
        batch_embeddings = embed_model.encode(batch_chunks).tolist()
        all_embeddings.extend(batch_embeddings)
        if progress_callback:
            progress_callback(min(i + BATCH, len(chunks)), len(chunks))

    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source_file": source_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, embeddings=all_embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


# --- UI ---
st.set_page_config(page_title="Транскрипты: суммаризация и RAG", layout="wide")
st.title("Транскрипты: суммаризация и RAG-поиск")

def get_project():
    """Селектор проекта в сайдбаре: выбор из существующих или ввод нового
    слага. Изолирует данные между клиентами — каждый проект своя папка
    data/<project>/ и своя коллекция ChromaDB."""
    existing = list_projects()
    options = existing + ["+ новый проект"]
    default = get_active_project()
    default_index = options.index(default) if default in existing else 0

    choice = st.sidebar.selectbox("Проект", options, index=default_index)
    if choice == "+ новый проект":
        choice = st.sidebar.text_input("Слаг нового проекта (буквы/цифры/-/_)")
        if not choice:
            st.sidebar.warning("Введи имя проекта, чтобы продолжить")
            st.stop()
    try:
        validate_project(choice)
    except ValueError as e:
        st.sidebar.error(str(e))
        st.stop()
    return choice


project = get_project()
api_key = get_api_key()
embed_model = load_embed_model()
collection = load_chroma_collection(project)

tab1, tab2 = st.tabs(["Обработать файл", "Спросить базу знаний"])

with tab1:
    st.subheader("Загрузка .docx транскриптов (можно сразу несколько)")
    uploaded_files = st.file_uploader(
        "Выбери файлы", type=["docx"], accept_multiple_files=True
    )

    if "parsed_files" not in st.session_state:
        st.session_state["parsed_files"] = {}

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state["parsed_files"]:
                segments = parse_docx(uf)
                full_text = " ".join(seg["text"] for seg in segments)
                st.session_state["parsed_files"][uf.name] = {
                    "segments": segments,
                    "full_text": full_text,
                    "summary": None,
                    "added_to_rag": False,
                }

        st.write(f"Загружено файлов: {len(uploaded_files)}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Суммаризировать все"):
                if not api_key:
                    st.error("Нужен API-ключ RouterAI (введи слева в панели)")
                else:
                    progress = st.progress(0.0, text="Суммаризация...")
                    names = [uf.name for uf in uploaded_files]
                    for i, name in enumerate(names):
                        entry = st.session_state["parsed_files"][name]
                        if entry["summary"] is None:
                            try:
                                entry["summary"] = summarize_transcript(entry["full_text"], api_key)
                            except Exception as e:
                                entry["summary"] = {"error": str(e)}
                        progress.progress((i + 1) / len(names), text=f"Обработано: {name}")
                    progress.empty()
                    st.success("Суммаризация всех файлов завершена")

        with col2:
            if st.button("Добавить все в RAG-базу"):
                progress = st.progress(0.0, text="Индексация...")
                names = [uf.name for uf in uploaded_files]
                added = 0
                for i, name in enumerate(names):
                    entry = st.session_state["parsed_files"][name]
                    if not entry["added_to_rag"]:
                        def update_progress(done, total, name=name):
                            progress.progress(
                                (i + done / total) / len(names),
                                text=f"{name}: чанк {done}/{total}"
                            )
                        add_to_rag(name, entry["full_text"], collection, embed_model, update_progress)
                        entry["added_to_rag"] = True
                        added += 1
                    progress.progress((i + 1) / len(names), text=f"Обработано: {name}")
                progress.empty()
                st.success(f"Добавлено в базу файлов: {added} (пропущено уже добавленных: {len(names) - added})")

        st.markdown("---")

        # Кнопка скачать все готовые summary одним файлом -
        # без неё результаты живут только в памяти сессии
        ready_summaries = {
            name: entry["summary"]
            for name, entry in st.session_state["parsed_files"].items()
            if entry["summary"] is not None and "error" not in entry["summary"]
        }
        if ready_summaries:
            combined = json.dumps(ready_summaries, ensure_ascii=False, indent=2)
            st.download_button(
                label=f"Скачать все summary ({len(ready_summaries)} шт.) в JSON",
                data=combined,
                file_name="all_summaries.json",
                mime="application/json",
            )

            readable_lines = []
            for name, s in ready_summaries.items():
                readable_lines.append(f"# {name}\n")
                readable_lines.append(f"{s['summary']}\n")
                readable_lines.append(f"Темы: {', '.join(s['themes'])}\n")
                readable_lines.append("Цитаты:")
                for q in s["quotes"]:
                    readable_lines.append(f"  - {q}")
                readable_lines.append("\n---\n")
            readable_text = "\n".join(readable_lines)

            st.download_button(
                label="Скачать все summary в читаемом виде (.txt)",
                data=readable_text,
                file_name="all_summaries.txt",
                mime="text/plain",
            )

        st.markdown("---")
        for uf in uploaded_files:
            entry = st.session_state["parsed_files"][uf.name]
            status = []
            if entry["summary"] is not None:
                status.append("summary готово")
            if entry["added_to_rag"]:
                status.append("в RAG-базе")
            status_text = f" ({', '.join(status)})" if status else ""

            with st.expander(f"{uf.name}{status_text}"):
                st.write(f"Найдено реплик: {len(entry['segments'])}")
                for seg in entry["segments"][:3]:
                    st.write(f"**{seg['speaker']}:** {seg['text'][:120]}")

                if entry["summary"]:
                    if "error" in entry["summary"]:
                        st.error(f"Ошибка суммаризации: {entry['summary']['error']}")
                    else:
                        result = entry["summary"]
                        st.markdown("**Summary:**")
                        st.write(result["summary"])
                        st.write("**Темы:**", ", ".join(result["themes"]))
                        st.write("**Цитаты:**")
                        for quote in result["quotes"]:
                            st.write(f"> {quote}")

with tab2:
    st.subheader("Вопрос по базе знаний")
    st.caption(f"Сейчас в базе: {collection.count()} чанков")

    question = st.text_input("Твой вопрос")
    if st.button("Спросить") and question:
        if not api_key:
            st.error("Нужен API-ключ RouterAI (введи слева в панели)")
        else:
            with st.spinner("Ищу и формирую ответ..."):
                answer = ask_knowledge_base(question, project, api_key=api_key)
            st.markdown("### Ответ")
            st.write(answer)
