"""
RAG Básico con ChromaDB persistente — Interfaz Streamlit (tema blanco · chatbot)
Replica el pipeline del notebook notebooks/04_rag/01_basic_rag.ipynb
"""

from __future__ import annotations

import os
from pathlib import Path
import warnings
import datetime
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant · ML Papers",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: tema blanco, estilo chatbot moderno ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f5f7fa; }

[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e8eaed;
}
[data-testid="stSidebar"] * { color: #1a1a2e !important; }

/* No ocultar el header ni la toolbar: ahí vive el botón del menú lateral */
header[data-testid="stHeader"] {
    background-color: transparent;
}
[data-testid="stDecoration"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"],
[data-testid="stMainMenuButton"],
.stDeployButton,
#MainMenu,
footer {
    display: none !important;
}
/* Botón para abrir/cerrar el panel de configuración (menú lateral) */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button {
    color: #374151 !important;
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 900px !important;
}

/* ── Header ── */
.app-header {
    display: flex; align-items: center; gap: 14px;
    padding: 22px 0 14px 0;
    border-bottom: 1px solid #e8eaed;
}
.header-avatar {
    width: 46px; height: 46px; border-radius: 14px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
}
.header-text h1 { font-size: 1.2rem; font-weight: 700; color: #111827; margin: 0; line-height: 1.3; }
.header-text p  { font-size: 0.78rem; color: #6b7280; margin: 0; }
.status-dot {
    width: 8px; height: 8px; background: #22c55e;
    border-radius: 50%; display: inline-block; margin-right: 5px;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ── Mensajes ── */
.chat-wrapper {
    padding: 20px 0 12px 0;
    min-height: 52vh; max-height: 58vh;
    overflow-y: auto; display: flex; flex-direction: column; gap: 4px;
}
.chat-wrapper::-webkit-scrollbar { width: 5px; }
.chat-wrapper::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 3px; }

.welcome-msg { text-align: center; padding: 40px 20px; color: #9ca3af; }
.welcome-msg .icon { font-size: 3rem; margin-bottom: 12px; }
.welcome-msg h3 { font-size: 1.1rem; font-weight: 600; color: #374151; margin-bottom: 6px; }
.welcome-msg p  { font-size: 0.85rem; line-height: 1.6; }

.msg-row {
    display: flex; align-items: flex-end; gap: 10px; margin: 6px 0;
    animation: fadeUp 0.25s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.msg-row.user { flex-direction: row-reverse; }

.msg-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; flex-shrink: 0;
}
.msg-avatar.ai-av  { background: linear-gradient(135deg,#6366f1,#8b5cf6); }
.msg-avatar.usr-av { background: linear-gradient(135deg,#0ea5e9,#06b6d4); }

.bubble {
    padding: 11px 16px; border-radius: 18px;
    font-size: 0.9rem; line-height: 1.65; max-width: 75%; word-wrap: break-word;
}
.bubble.ai {
    background: #ffffff; color: #1f2937;
    border: 1px solid #e5e7eb; border-bottom-left-radius: 5px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.bubble.user {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff; border-bottom-right-radius: 5px;
    box-shadow: 0 2px 10px rgba(99,102,241,0.25);
}

.ts {
    font-size: 0.68rem; color: #9ca3af;
    padding: 0 4px; align-self: flex-end; margin-bottom: 2px; white-space: nowrap;
}

/* ── Input ── */
.input-bar {
    border-top: 1px solid #e8eaed; background: #ffffff;
    padding: 14px 0 8px 0;
}
.stTextInput > label { display: none !important; }
.stTextInput > div > div > input {
    background: #f9fafb !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 14px !important;
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #9ca3af !important; }

/* ── Botones ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 0.88rem !important; padding: 10px 18px !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 3px 10px rgba(99,102,241,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover  { opacity: 0.88 !important; }
.stButton > button:active { transform: scale(0.97) !important; }

/* ── Sidebar ── */
.sidebar-section {
    background: #f9fafb; border: 1px solid #e8eaed; border-radius: 12px;
    padding: 14px; margin: 10px 0; font-size: 0.82rem; color: #374151;
}
.sidebar-section strong { color: #6366f1; }

.stat-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin: 3px 3px 3px 0;
}
.chip-green { background: #dcfce7; color: #16a34a; }
.chip-blue  { background: #dbeafe; color: #2563eb; }
.chip-gray  { background: #f3f4f6; color: #6b7280; }
.chip-red   { background: #fee2e2; color: #dc2626; }

/* ── Expander ── */
.stExpander {
    border: 1px solid #e8eaed !important;
    border-radius: 10px !important;
    background: #fff !important; margin-top: 6px !important;
}

</style>
""", unsafe_allow_html=True)

# ── Constantes ──────────────────────────────────────────────────────────────
BASE_DIR   = str(Path(__file__).resolve().parents[2])
PDF_DIR    = os.path.join(BASE_DIR, "data", "Papers")
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_db")

PAPER_URLS = [
    "https://arxiv.org/pdf/2306.06031v1.pdf",
    "https://arxiv.org/pdf/2306.12156v1.pdf",
    "https://arxiv.org/pdf/2306.14289v1.pdf",
    "https://arxiv.org/pdf/2305.10973v1.pdf",
    "https://arxiv.org/pdf/2306.13643v1.pdf",
]

# Modelo de embeddings fijo
EMB_MODEL = "nomic-embed-text:latest"

# Modelos LLM disponibles
LLM_MODELS = ["qwen2.5:1.5b", "llama3.2:3b", "llama3.2:latest", "llama3.2:1b"]

# ── Estado de sesión ────────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "chain": None,
    "vectorstore": None,
    "retriever": None,
    "last_retrieved_chunks": [],
    "pending_q": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers de backend ──────────────────────────────────────────────────────
def download_papers():
    os.makedirs(PDF_DIR, exist_ok=True)
    for i, url in enumerate(PAPER_URLS):
        fname = os.path.join(PDF_DIR, f"paper{i+1}.pdf")
        if not os.path.exists(fname):
            r = requests.get(url, timeout=60)
            with open(fname, "wb") as f:
                f.write(r.content)


def get_existing_embedding_dim() -> int | None:
    """Lee la dimensión de embeddings almacenada en ChromaDB sin invocar el modelo."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTOR_DIR)
        col  = client.get_collection("ml_papers")
        peek = col.peek(limit=1)
        if peek and peek.get("embeddings") and peek["embeddings"]:
            return len(peek["embeddings"][0])
    except Exception:
        pass
    return None


def wipe_vector_db():
    """Elimina el directorio de ChromaDB y limpia la caché de Streamlit."""
    import shutil
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    build_rag_chain.clear()   # invalida el cache_resource


def _chunk_dict(doc, score: float | None = None) -> dict:
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("source", "?"),
        "page": doc.metadata.get("page", "?"),
        "score": score,
        "chars": len(doc.page_content),
    }


def fetch_retrieved_chunks(vectorstore, retriever, query: str, k: int) -> list[dict]:
    """Recupera los mismos fragmentos que usa el RAG, con puntuación de similitud."""
    try:
        scored = vectorstore.similarity_search_with_score(query, k=k)
        if scored:
            return [_chunk_dict(doc, float(score)) for doc, score in scored]
    except Exception:
        pass

    docs = retriever.invoke(query)
    return [_chunk_dict(doc) for doc in docs]


def render_retrieved_chunks(chunks: list[dict], turn_id: int) -> None:
    """Muestra el texto completo extraído de la base vectorial."""
    if not chunks:
        st.caption("No se encontraron fragmentos relevantes.")
        return

    total_chars = sum(c["chars"] for c in chunks)
    st.caption(
        f"{len(chunks)} fragmento(s) recuperados · "
        f"{total_chars:,} caracteres enviados como contexto al LLM"
    )

    for i, chunk in enumerate(chunks, 1):
        src = os.path.basename(str(chunk["source"]))
        score = chunk.get("score")
        score_label = f" · similitud {score:.4f}" if score is not None else ""
        label = f"📄 Fragmento {i} — {src} (pág. {chunk['page']}){score_label}"

        with st.expander(label, expanded=(i == 1)):
            st.markdown(
                f"**Origen:** `{chunk['source']}` · "
                f"**Página:** {chunk['page']} · "
                f"**Longitud:** {chunk['chars']:,} caracteres"
            )
            height = min(420, max(140, chunk["chars"] // 3))
            st.text_area(
                "Texto extraído",
                value=chunk["content"],
                height=height,
                disabled=True,
                label_visibility="collapsed",
                key=f"chunk_{turn_id}_{i}",
            )


@st.cache_resource(show_spinner=False)
def build_rag_chain(emb_model: str, llm_model: str, top_k: int, temperature: float):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    embeddings = OllamaEmbeddings(model=emb_model)

    if os.path.exists(VECTOR_DIR) and os.listdir(VECTOR_DIR):
        vectorstore = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings,
            collection_name="ml_papers",
        )
    else:
        download_papers()
        ml_papers = []
        for i in range(len(PAPER_URLS)):
            loader = PyPDFLoader(os.path.join(PDF_DIR, f"paper{i+1}.pdf"))
            ml_papers.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(ml_papers)
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=VECTOR_DIR,
            collection_name="ml_papers",
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    llm = ChatOllama(model=llm_model, temperature=temperature)

    prompt = PromptTemplate.from_template("""
Eres un asistente experto en inteligencia artificial y aprendizaje automático.
Responde SIEMPRE en español, de forma clara y detallada, usando el contexto proporcionado.
Si el contexto no contiene suficiente información para responder, indícalo con amabilidad.

Contexto:
{context}

Pregunta:
{question}

Respuesta en español:""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {
            "context":  RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt | llm | StrOutputParser()
    )
    return chain, vectorstore, retriever


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    emb_model   = EMB_MODEL  # Fijo: nomic-embed-text:latest
    llm_model   = st.selectbox("Modelo LLM", LLM_MODELS)
    top_k       = st.slider("Chunks (k)", 1, 10, 5)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.0, 0.05)
    show_src    = st.toggle("Mostrar fuentes", value=True)

    st.markdown("---")

    if st.button("🗑️ Nueva conversación", use_container_width=True, key="clear-btn"):
        st.session_state.messages = []
        st.session_state.last_retrieved_chunks = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INICIALIZAR RAG (una sola vez por sesión)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.chain is None or st.session_state.retriever is None:
    with st.spinner("⚙️ Inicializando el sistema RAG…"):
        try:
            chain, vs, retriever = build_rag_chain(emb_model, llm_model, top_k, temperature)
            st.session_state.chain       = chain
            st.session_state.vectorstore = vs
            st.session_state.retriever   = retriever
        except Exception as e:
            st.error(f"❌ Error al inicializar el sistema RAG:\n\n{e}")
            st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
  <div class="header-avatar">💬</div>
  <div class="header-text">
    <h1>RAG Assistant</h1>
    <p><span class="status-dot"></span>En línea</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORIAL DE MENSAJES
# ══════════════════════════════════════════════════════════════════════════════
def render_messages():
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-msg">
          <div class="icon">🤖</div>
          <h3>¡Hola! Soy tu asistente RAG</h3>
          <p>Puedo responder preguntas sobre los papers de ML e IA indexados.<br/>
          Usa los accesos rápidos de abajo o escribe tu pregunta directamente.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    html = '<div class="chat-wrapper">'
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"].replace("\n", "<br/>")
        ts      = msg.get("ts", "")
        if role == "user":
            html += f"""
            <div class="msg-row user">
              <span class="ts">{ts}</span>
              <div class="bubble user">{content}</div>
              <div class="msg-avatar usr-av">🧑</div>
            </div>"""
        else:
            html += f"""
            <div class="msg-row ai">
              <div class="msg-avatar ai-av">🤖</div>
              <div class="bubble ai">{content}</div>
              <span class="ts">{ts}</span>
            </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


render_messages()

# ── Fragmentos recuperados del último turno ────────────────────────────────
if show_src and st.session_state.last_retrieved_chunks:
    n_chunks = len(st.session_state.last_retrieved_chunks)
    turn_id = sum(1 for m in st.session_state.messages if m["role"] == "user")
    with st.expander(f"📚 Ver fragmentos recuperados ({n_chunks})", expanded=False):
        with st.container(height=340):
            try:
                render_retrieved_chunks(st.session_state.last_retrieved_chunks, turn_id)
            except Exception as src_err:
                if "dimension" in str(src_err).lower():
                    st.error(
                        "⚠️ **Mismatch de dimensiones en ChromaDB.**\n\n"
                        "El modelo de embeddings no coincide con el índice actual. "
                        "Reinicia la sesión o recrea el índice vectorial."
                    )
                else:
                    st.error(f"Error al mostrar fragmentos: {src_err}")


# ══════════════════════════════════════════════════════════════════════════════
# BARRA DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="input-bar">', unsafe_allow_html=True)

pending = st.session_state.pending_q
if pending:
    st.session_state.pending_q = ""

col_in, col_btn = st.columns([6, 1])
with col_in:
    user_input = st.text_input(
        "Pregunta",
        value=pending,
        placeholder="Escribe tu pregunta sobre los papers…",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Enviar ↑", use_container_width=True, key="send_btn")

st.markdown("</div>", unsafe_allow_html=True)


# ── Procesar consulta ───────────────────────────────────────────────────────
if send and user_input.strip():
    q  = user_input.strip()
    ts = datetime.datetime.now().strftime("%H:%M")

    st.session_state.messages.append({"role": "user", "content": q, "ts": ts})
    st.session_state.pending_q = ""  # limpia el campo de texto al hacer rerun

    with st.spinner("Pensando…"):
        try:
            st.session_state.last_retrieved_chunks = fetch_retrieved_chunks(
                st.session_state.vectorstore,
                st.session_state.retriever,
                q,
                top_k,
            )
            answer = st.session_state.chain.invoke({"question": q})
        except Exception as e:
            st.session_state.last_retrieved_chunks = []
            err = str(e)
            if "dimension" in err.lower():
                answer = (
                    "⚠️ Error de dimensiones: el modelo de embeddings activo no coincide "
                    "con el índice almacenado. Reinicia la sesión o contacta al administrador."
                )
            else:
                answer = f"⚠️ Error al generar respuesta: {e}"

    ts2 = datetime.datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "assistant", "content": answer, "ts": ts2})
    st.rerun()
