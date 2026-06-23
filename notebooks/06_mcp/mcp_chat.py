#!/usr/bin/env python3
"""
Chat de consola para explicar MCP con OpenAI u Ollama.

Uso:
  python notebooks/06_mcp/mcp_chat.py
  python notebooks/06_mcp/mcp_chat.py --provider openai --model gpt-4o-mini
  python notebooks/06_mcp/mcp_chat.py --provider ollama --model llama3.1
"""

from __future__ import annotations

import json
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class Color:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"      # Entrada usuario
    GREEN = "\033[92m"     # Salida asistente
    YELLOW = "\033[93m"    # Estado/acciones MCP
    MAGENTA = "\033[95m"   # Titulo/headers
    RED = "\033[91m"       # Errores


Role = Literal["system", "user", "assistant"]


class ChatTurn(TypedDict):
    role: Role
    content: str


@dataclass
class ConsoleMCPChat:
    provider: Literal["openai", "ollama"] = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.2
    history_file: Path = Path("notebooks/06_mcp/mcp_chat_history.json")
    max_context_turns: int = 12

    def __post_init__(self) -> None:
        if self.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        else:
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=self.temperature,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )
        self.history: List[ChatTurn] = []
        self.weather_data = {
            "bogota": 18,
            "medellin": 24,
            "mexico": 21,
        }
        self._load_history()

    def _load_history(self) -> None:
        if self.history_file.exists():
            try:
                self.history = json.loads(self.history_file.read_text(encoding="utf-8"))
                print(
                    f"{Color.YELLOW}[MCP] Historial cargado: "
                    f"{len(self.history)} mensajes.{Color.RESET}"
                )
            except (json.JSONDecodeError, OSError):
                print(
                    f"{Color.RED}[MCP] No se pudo cargar historial, "
                    "se inicia vacío." + Color.RESET
                )
                self.history = []

    def _save_history(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.history_file.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _mcp_trace(self) -> None:
        print(f"{Color.YELLOW}[MCP] 1) Entender pregunta del usuario.{Color.RESET}")
        print(f"{Color.YELLOW}[MCP] 2) Revisar historial reciente.{Color.RESET}")
        print(f"{Color.YELLOW}[MCP] 3) Detectar si requiere tool de clima.{Color.RESET}")
        print(f"{Color.YELLOW}[MCP] 4) Si aplica, ejecutar get_weather.{Color.RESET}")
        print(
            f"{Color.YELLOW}[MCP] 5) (Opcional) Consultar modelo ({self.provider}:{self.model_name})."
            f"{Color.RESET}"
        )
        print(f"{Color.YELLOW}[MCP] 6) Responder al usuario.{Color.RESET}")

    def _is_weather_query(self, text: str) -> bool:
        lowered = text.lower()
        return "clima" in lowered or "temperatura" in lowered

    def _extract_city(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for city in self.weather_data:
            if city in lowered:
                return city
        return None

    def _extract_unit(self, text: str) -> str:
        lowered = text.lower()
        if " fahrenheit" in lowered or " f" in lowered:
            return "F"
        return "C"

    def get_weather(self, city: str, unit: str = "C") -> dict:
        temp_c = self.weather_data.get(city.lower())
        if temp_c is None:
            return {"ok": False, "error": "Ciudad no encontrada"}

        if unit == "F":
            temp = (temp_c * 9 / 5) + 32
        else:
            temp = temp_c

        return {
            "ok": True,
            "city": city,
            "temperature": round(temp, 1),
            "unit": unit,
        }

    def _build_messages(self, user_input: str):
        system_prompt = (
            "Eres un tutor de MCP (Model Context Protocol). "
            "Explica en espanol simple, con ejemplos cortos y paso a paso. "
            "Si el usuario pregunta algo tecnico, responde claro y practico."
        )

        recent_history = self.history[-self.max_context_turns :]
        messages = [SystemMessage(content=system_prompt)]
        for turn in recent_history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_input))
        return messages

    def ask(self, user_input: str) -> str:
        self._mcp_trace()

        if self._is_weather_query(user_input):
            city = self._extract_city(user_input)
            if not city:
                answer = "Puedo consultar clima, pero necesito que indiques una ciudad."
            else:
                unit = self._extract_unit(user_input)
                tool_result = self.get_weather(city, unit=unit)
                print(
                    f"{Color.YELLOW}[MCP] Tool get_weather({city}, {unit}) -> {tool_result}"
                    f"{Color.RESET}"
                )
                if not tool_result["ok"]:
                    answer = f"No pude obtener el clima: {tool_result['error']}"
                else:
                    answer = (
                        f"En {tool_result['city'].title()} hay {tool_result['temperature']} "
                        f"grados {tool_result['unit']}."
                    )
        else:
            answer = "Puedo ayudarte con preguntas de clima."

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": answer})
        self._save_history()
        return answer


def print_welcome(provider: str, model_name: str) -> None:
    print(f"{Color.MAGENTA}{Color.BOLD}=== MCP Chat Console ==={Color.RESET}")
    print(f"{Color.YELLOW}Caso de uso: clima (tool simulada get_weather){Color.RESET}")
    print(f"{Color.YELLOW}Proveedor: {provider} | Modelo: {model_name}{Color.RESET}")
    print(f"{Color.YELLOW}Comandos: /salir, /historia, /limpiar{Color.RESET}")
    print(
        f"{Color.YELLOW}Colores -> Entrada: cian | Salida: verde | MCP: amarillo{Color.RESET}"
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Chat MCP de consola")
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=os.getenv("MCP_PROVIDER", "openai"),
        help="Proveedor de modelo (openai u ollama)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Nombre del modelo. Si no se indica, usa uno por defecto por proveedor.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura del modelo",
    )
    args = parser.parse_args()

    default_model = "gpt-4o-mini" if args.provider == "openai" else "llama3.1"
    model_name = args.model or default_model

    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print(
            f"{Color.RED}No se encontro OPENAI_API_KEY. "
            "Agregala en .env o variables de entorno."
            f"{Color.RESET}"
        )
        return

    chat = ConsoleMCPChat(
        provider=args.provider,
        model_name=model_name,
        temperature=args.temperature,
    )
    print_welcome(args.provider, model_name)

    while True:
        user_input = input(f"{Color.CYAN}{Color.BOLD}Tu pregunta > {Color.RESET}").strip()

        if not user_input:
            continue
        if user_input.lower() == "/salir":
            print(f"{Color.MAGENTA}Hasta luego.{Color.RESET}")
            break
        if user_input.lower() == "/historia":
            print(f"{Color.YELLOW}[MCP] Mensajes guardados: {len(chat.history)}{Color.RESET}")
            continue
        if user_input.lower() == "/limpiar":
            chat.history = []
            chat._save_history()
            print(f"{Color.YELLOW}[MCP] Historial limpiado.{Color.RESET}")
            continue

        try:
            answer = chat.ask(user_input)
            print(f"{Color.GREEN}{Color.BOLD}Respuesta > {answer}{Color.RESET}")
        except Exception as exc:  # noqa: BLE001
            print(f"{Color.RED}Error consultando {args.provider}: {exc}{Color.RESET}")


if __name__ == "__main__":
    main()
