from __future__ import annotations

from functools import lru_cache
from typing import Literal

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
import google.api_core.exceptions as google_exceptions

from src.logs import get_logger, short_text
from src.rag.RAG import consumir_rag_mundiales
from src.tools.tools import get_worldcup_tools
from src.utils.moderador import is_inappropriate_input

'''
def obtener_llm(temperature: float = 0):
    # Instancia fallback local (Ollama)
    ollama_llm = ChatOllama(model="llama3.2:3b", temperature=temperature)
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        # Instancia Gemini
        gemini_llm = ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            temperature=temperature
        )
        # Configura fallback Ollama si falla Gemini
        return gemini_llm.with_fallbacks([ollama_llm])
    
    # Si no hay API key, usa Ollama directamente
    return ollama_llm
'''

def obtener_llm(temperature: float = 0):
    ollama_llm = ChatOllama(model="llama3.2:3b", temperature=temperature)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            gemini_llm = ChatGoogleGenerativeAI(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                temperature=temperature
            )
            return gemini_llm.with_fallbacks([ollama_llm])

        except google_exceptions.Unauthorized as e:
            logger.warning("API Key inválida o sin permisos: %s", e)
        except google_exceptions.ResourceExhausted as e:
            logger.warning("API Key válida pero sin créditos/cuota: %s", e)
        except Exception as e:
            logger.warning("Error inesperado inicializando Gemini: %s", e)

        logger.info("Usando Ollama como fallback.")
        return ollama_llm

    logger.info("No se encontró GOOGLE_API_KEY, usando Ollama.")
    return ollama_llm


try:
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

DEFAULT_MODEL_NAME = "llama3.2:3b"
MEMORY_WINDOW = 5
logger = get_logger("worldcup_graph")


class GraphState(TypedDict):
    user_input: str
    chat_history: list[dict[str, str]]
    route: Literal["rag", "tools", "chat", "blocked"]
    output: str


def _history_as_text(chat_history: list[dict[str, str]], last_n: int = MEMORY_WINDOW) -> str:
    recent = chat_history[-last_n:]
    if not recent:
        return "Sin historial."
    return "\n".join(
        f"{msg.get('role', 'user')}: {msg.get('content', '')}".strip()
        for msg in recent
    )


def _history_as_messages(chat_history: list[dict[str, str]], last_n: int = MEMORY_WINDOW):
    msgs = []
    for msg in chat_history[-last_n:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            msgs.append(AIMessage(content=content))
        else:
            msgs.append(HumanMessage(content=content))
    return msgs


def _execute_tool(tool_obj, args: dict):
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(args)
    return tool_obj(**args)


@lru_cache(maxsize=2)
def _router_llm(model_name: str):
    #return ChatOllama(model=model_name, temperature=0)
    return obtener_llm(0)


@lru_cache(maxsize=2)
def _tools_llm(model_name: str):
    #return ChatOllama(model=model_name, temperature=0)
    return obtener_llm(0)


@lru_cache(maxsize=2)
def _chat_llm(model_name: str):
    #return ChatOllama(model=model_name, temperature=0.2)
    return obtener_llm(0.2)


@lru_cache(maxsize=1)
def _tool_map():
    tools = get_worldcup_tools()
    return {
        getattr(t, "name", getattr(t, "__name__", f"tool_{i}")): t
        for i, t in enumerate(tools)
    }


def _guardrails_node(state: GraphState) -> GraphState:
    if is_inappropriate_input(state["user_input"]):
        logger.info("Guardrails bloqueo mensaje: %s", short_text(state["user_input"]))
        return {
            **state,
            "route": "blocked",
            "output": "Consulta bloqueada por guardrails: lenguaje inapropiado detectado.",
        }
    logger.info("Guardrails aprobado mensaje: %s", short_text(state["user_input"]))
    return {**state, "route": "rag", "output": ""}


def _router_node(state: GraphState, model_name: str) -> GraphState:
    text = state["user_input"]
    history_text = _history_as_text(state.get("chat_history", []))
    prompt = f"""
Eres un enrutador para un asistente de Mundiales.
Decide SOLO una opcion:
- tools: cuando pidan metricas, estadisticas, top, promedio, conteos o analisis tabular.
- rag: cuando pidan historia, contexto narrativo, campeones, hechos o informacion documental.
- chat: cuando sea saludo, agradecimiento, despedida o conversacion social breve.

Historial reciente:
{history_text}

Pregunta del usuario: {text}

Responde exactamente una palabra: tools, rag o chat
""".strip()

    decision = (_router_llm(model_name).invoke(prompt).content or "").strip().lower()
    if "chat" in decision:
        route = "chat"
    elif "tools" in decision:
        route = "tools"
    else:
        route = "rag"
    logger.info(
        "Router decision=%s | user_input=%s",
        route,
        short_text(text),
    )
    return {**state, "route": route}


def _rag_node(state: GraphState, model_name: str) -> GraphState:
    logger.info("RAG start | input=%s", short_text(state["user_input"]))
    result = consumir_rag_mundiales(
        state["user_input"],
        llm_model=model_name,
        top_k=4,
        chat_history=state.get("chat_history", []),
        history_window=MEMORY_WINDOW,
    )
    fuentes = result.get("fuentes", [])[:2]
    fuentes_text = "\n".join(
        f"- {f.get('source', '?')} (pag. {f.get('page', 'N/A')})"
        for f in fuentes
    )
    salida = (
        f"[RAG]\n{result.get('respuesta', 'Sin respuesta')}\n\n"
        f"Fuentes (max 2):\n{fuentes_text if fuentes_text else '- Sin fuentes'}"
    )
    logger.info(
        "RAG end | chunks=%s | output=%s",
        result.get("chunks", 0),
        short_text(result.get("respuesta", "")),
    )
    return {**state, "output": salida}


def _tools_node(state: GraphState, model_name: str) -> GraphState:
    logger.info("TOOLS start | input=%s", short_text(state["user_input"]))
    system_msg = (
        "Eres un analista amable de datos de partidos de mundiales. "
        "Usa tools cuando necesites datos exactos y responde en espanol con tono cordial, "
        "claro y breve."
    )
    tool_map = _tool_map()
    llm_with_tools = _tools_llm(model_name).bind_tools(list(tool_map.values()))
    messages = [
        ("system", system_msg),
        *_history_as_messages(state.get("chat_history", [])),
        HumanMessage(content=state["user_input"]),
    ]

    final_answer = ""
    for _ in range(3):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            final_answer = ai_msg.content or "Sin respuesta del agente de tools."
            break

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            tool_obj = tool_map.get(tool_name)
            if tool_obj is None:
                result = f"Tool no encontrada: {tool_name}"
            else:
                try:
                    result = _execute_tool(tool_obj, tool_args)
                    logger.info(
                        "Tool executed | name=%s | args=%s",
                        tool_name,
                        str(tool_args),
                    )
                except Exception as exc:
                    result = f"Error ejecutando {tool_name}: {exc}"
                    logger.exception("Tool execution error | name=%s", tool_name)

            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    if not final_answer:
        final_answer = "No se logro generar respuesta final con tools."
    logger.info("TOOLS end | output=%s", short_text(final_answer))
    return {**state, "output": f"[TOOLS]\n{final_answer}"}


def _chat_node(state: GraphState, model_name: str) -> GraphState:
    logger.info("CHAT start | input=%s", short_text(state["user_input"]))
    history_text = _history_as_text(state.get("chat_history", []))
    prompt = f"""
Eres un asistente amable especializado en Mundiales de Futbol.
Responde de forma cordial y breve (maximo 3 lineas).
Si el usuario saluda, saluda y ofrece ayuda concreta sobre mundiales.
Si agradece o se despide, responde con cortesia.

Historial reciente:
{history_text}

Mensaje del usuario:
{state["user_input"]}
""".strip()
    msg = _chat_llm(model_name).invoke(prompt)
    content = msg.content if hasattr(msg, "content") else str(msg)
    logger.info("CHAT end | output=%s", short_text(content))
    return {**state, "output": f"[CHAT]\n{content}"}


def _route_after_guard(state: GraphState):
    return END if state["route"] == "blocked" else "router"


def _route_after_router(state: GraphState):
    if state["route"] == "tools":
        return "tools"
    if state["route"] == "chat":
        return "chat"
    return "rag"


class _SimpleCompiledGraph:
    """Fallback cuando langgraph no esta instalado."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, state: GraphState) -> GraphState:
        s = _guardrails_node(state)
        if s["route"] == "blocked":
            return s
        s = _router_node(s, self.model_name)
        if s["route"] == "tools":
            return _tools_node(s, self.model_name)
        if s["route"] == "chat":
            return _chat_node(s, self.model_name)
        return _rag_node(s, self.model_name)


@lru_cache(maxsize=2)
def build_worldcup_graph(model_name: str = DEFAULT_MODEL_NAME):
    if not LANGGRAPH_AVAILABLE:
        return _SimpleCompiledGraph(model_name=model_name)

    builder = StateGraph(GraphState)
    builder.add_node("guardrails", _guardrails_node)
    builder.add_node("router", lambda s: _router_node(s, model_name))
    builder.add_node("rag", lambda s: _rag_node(s, model_name))
    builder.add_node("tools", lambda s: _tools_node(s, model_name))
    builder.add_node("chat", lambda s: _chat_node(s, model_name))

    builder.add_edge(START, "guardrails")
    builder.add_conditional_edges(
        "guardrails",
        _route_after_guard,
        {"router": "router", END: END},
    )
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {"rag": "rag", "tools": "tools", "chat": "chat"},
    )
    builder.add_edge("rag", END)
    builder.add_edge("tools", END)
    builder.add_edge("chat", END)
    return builder.compile()


def ask_worldcup_graph(
    question: str,
    chat_history: list[dict[str, str]] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> str:
    logger.info("ask_worldcup_graph start | input=%s", short_text(question))
    graph = build_worldcup_graph(model_name=model_name)
    state_in: GraphState = {
        "user_input": question,
        "chat_history": (chat_history or [])[-MEMORY_WINDOW:],
        "route": "rag",
        "output": "",
    }
    out = graph.invoke(state_in)
    logger.info("ask_worldcup_graph end | output=%s", short_text(out["output"]))
    return out["output"]
