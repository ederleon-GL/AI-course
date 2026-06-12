#!/usr/bin/env python3
"""
Chatbot de consola con memoria LangChain y OpenAI.
Uso: python chat_memory_console.py
"""

import logging
import os
import sys
from pathlib import Path

# Silenciar aviso de transformers/PyTorch (no usamos modelos HF en este script)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
logging.getLogger("transformers").setLevel(logging.CRITICAL)

import tiktoken
from dotenv import load_dotenv
from langchain_core._api.deprecation import suppress_langchain_deprecation_warning
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryBufferMemory,
    ConversationSummaryMemory,
)
from langchain_openai import ChatOpenAI

# Cargar OPENAI_API_KEY desde la raíz del proyecto
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL = "gpt-4o-mini"
RESET = "\033[0m"
TURN_COLORS = ["\033[36m", "\033[32m", "\033[33m", "\033[35m", "\033[34m", "\033[91m"]


def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))


def memory_text(memory) -> str:
    """Texto que la memoria inyectará en el prompt."""
    data = memory.load_memory_variables({})
    history = data.get("history", "")
    if isinstance(history, list):
        return "\n".join(getattr(m, "content", str(m)) for m in history)
    return str(history)


def build_memory(choice: str, llm: ChatOpenAI):
    options = {
        "1": lambda: ConversationBufferMemory(),
        "2": lambda: ConversationBufferWindowMemory(k=4),
        "3": lambda: ConversationSummaryMemory(llm=llm),
        "4": lambda: ConversationSummaryBufferMemory(llm=llm, max_token_limit=500),
    }
    return options[choice]()


def select_memory(llm: ChatOpenAI):
    print("\nSelecciona el tipo de memoria:")
    print("  1) Buffer          — historial completo")
    print("  2) Ventana (k=4)   — últimas 4 interacciones")
    print("  3) Resumen         — resumen progresivo con LLM")
    print("  4) Resumen+Buffer  — recientes en bruto + resumen")
    while True:
        choice = input("\nOpción [1-4]: ").strip()
        if choice in {"1", "2", "3", "4"}:
            return build_memory(choice, llm)
        print("Opción no válida.")


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: define OPENAI_API_KEY en .env")
        sys.exit(1)

    llm = ChatOpenAI(model=MODEL, temperature=0.3)

    with suppress_langchain_deprecation_warning():
        memory = select_memory(llm)
        chat = ConversationChain(llm=llm, memory=memory, verbose=False)

    print(f"\nChat iniciado ({MODEL}). Escribe 'salir' para terminar.\n")

    turn = 0
    with suppress_langchain_deprecation_warning():
        while True:
            user = input("Tú: ").strip()
            if not user:
                continue
            if user.lower() in {"salir", "exit", "quit"}:
                print("Hasta luego.")
                break

            turn += 1
            reply = chat.predict(input=user)
            tokens = count_tokens(memory_text(memory))
            color = TURN_COLORS[(turn - 1) % len(TURN_COLORS)]

            print(f"Asistente: {reply}")
            print(f"{color}📊 Memoria → {tokens} tokens en este turno{RESET}\n")


if __name__ == "__main__":
    main()
