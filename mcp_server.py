"""
mcp_server.py

Минимальный MCP-сервер поверх RAG-проекта — один инструмент, search_transcripts,
который ищет по той же ChromaDB, что и rag.py/graph_rag.py, но отдаёт сырые
найденные чанки с указанием источника, а не сгенерированный LLM-ответ.
Генерация и рассуждение по найденному — задача клиента (Claude Desktop),
а не сервера; сервер — просто инструмент поиска.

Использование:
    Добавить в claude_desktop_config.json (Windows — обычно
    %APPDATA%\\Claude\\claude_desktop_config.json):

    {
      "mcpServers": {
        "rag-transcripts": {
          "command": "C:\\путь\\к\\проекту\\venv\\Scripts\\python.exe",
          "args": ["C:\\путь\\к\\проекту\\mcp_server.py"]
        }
      }
    }

    Затем перезапустить Claude Desktop.

Требования:
    pip install mcp --break-system-packages
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer as FastMCP

from config import get_active_project
from indexing import get_collection, get_embed_model

mcp = FastMCP("rag-transcripts")


@mcp.tool()
def search_transcripts(query: str, top_k: int = 5, project: str = "") -> str:
    """Ищет релевантные фрагменты интервью в базе знаний по семантическому запросу.

    Args:
        query: поисковый запрос на естественном языке
        top_k: сколько фрагментов вернуть (по умолчанию 5)
        project: имя проекта/коллекции; если пусто — берётся активный проект

    Returns:
        Найденные фрагменты текста с указанием источника (имени файла интервью).
    """
    project_name = project or get_active_project()
    embed_model = get_embed_model()
    collection = get_collection(project_name)

    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    if not results["documents"][0]:
        return "Ничего не найдено по данному запросу."

    blocks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source_name = Path(meta["source_file"]).stem
        blocks.append(f"[Источник: {source_name}]\n{doc}")

    return "\n\n---\n\n".join(blocks)


if __name__ == "__main__":
    mcp.run()
