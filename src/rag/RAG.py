from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.logs import get_logger, short_text

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text:latest"
DEFAULT_LLM_MODEL = "llama3.2:3b"
DEFAULT_COLLECTION = "mundiales_football"
DEFAULT_TOP_K = 10 #4
UMBRAL=0.6
logger = get_logger("worldcup_rag")


def _project_root() -> Path:
    """Retorna la raiz del proyecto AI-course."""
    return Path(__file__).resolve().parents[2]


def _default_vector_dir() -> Path:
    return _project_root() / "data" / "vector_db_2"


@lru_cache(maxsize=4)
def _load_vectorstore(
    vector_dir: str,
    collection_name: str,
    embedding_model: str,
) -> Chroma:
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma(
        persist_directory=vector_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def consumir_rag_mundiales(
    pregunta: str,
    *,
    llm_model: str = DEFAULT_LLM_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = DEFAULT_TOP_K,
    vector_dir: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    history_window: int = 5,
) -> dict[str, Any]:
    """
    Consume la RAG de mundiales y retorna respuesta con fuentes.

    Returns:
        {
            "respuesta": str,
            "fuentes": [{"source": str, "page": int | str}],
            "chunks": int
        }
    """
    if not pregunta or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacia.")
    logger.info("consumir_rag_mundiales start | pregunta=%s", short_text(pregunta))

    db_dir = Path(vector_dir) if vector_dir else _default_vector_dir()
    if not db_dir.exists():
        raise FileNotFoundError(
            f"No existe la base vectorial en: {db_dir}. "
            "Genera el indice antes de consumir la RAG."
        )

    vectorstore = _load_vectorstore(
        str(db_dir),
        collection_name,
        embedding_model,
    )
    docs_and_scores = vectorstore.similarity_search_with_score(pregunta, k=top_k) #DEVUELVE SCORES TMB
    docs = [doc for doc, score in docs_and_scores if score >= UMBRAL] #filtra por umbral, si la similitud es menor a UMBRAL, no se incluye
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    ##docs = retriever.invoke(pregunta)

    

    contexto = "\n\n".join(doc.page_content for doc in docs)
    if not contexto.strip():
        logger.info("consumir_rag_mundiales end | sin fragmentos")
        return {"respuesta": "No se encontraron fragmentos relevantes.", "fuentes": [], "chunks": 0}

    recent_history = (chat_history or [])[-history_window:]
    history_text = "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}".strip()
        for msg in recent_history
    )
    if not history_text:
        history_text = "Sin historial previo."

    prompt = f"""
Eres un asistente experto en historia de los Mundiales de Futbol.
Responde solo con el contexto recuperado.
Si la informacion no es suficiente, dilo explicitamente.
Mantén un tono amable, claro y breve.

Historial reciente:
{history_text}

Contexto:
{contexto}

Pregunta: {pregunta}
Respuesta en espanol:
""".strip()

    llm = ChatOllama(model=llm_model, temperature=0)
    respuesta_llm = llm.invoke(prompt)
    respuesta = respuesta_llm.content if hasattr(respuesta_llm, "content") else str(respuesta_llm)

    fuentes = [
        {
            "source": doc.metadata.get("source", "desconocido"),
            "page": doc.metadata.get("page", "N/A"),
        }
        for doc in docs
    ]

    output = {
        "respuesta": respuesta,
        "fuentes": fuentes,
        "chunks": len(docs),
    }
    logger.info(
        "consumir_rag_mundiales end | chunks=%s | respuesta=%s",
        len(docs),
        short_text(respuesta),
    )
    return output
