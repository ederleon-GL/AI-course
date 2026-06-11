# setup_ollama_local.py
# ------------------------------------------------------------------
# Pure‑local Ollama setup – no Google Colab detection
# ------------------------------------------------------------------

import os
import subprocess
import time
import shutil
import socket


def start_ollama_server() -> bool:
    """Start Ollama in the background and wait until it answers on port 11434."""
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Poll the local port until the server is ready (max 30 seconds)
        for _ in range(30):
            try:
                s = socket.socket()
                s.settimeout(1)
                s.connect(("127.0.0.1", 11434))
                s.close()
                print("✅ Servidor Ollama listo en el puerto 11434!")
                return True
            except Exception:
                time.sleep(1)
        print("❌ Error: el servidor Ollama no respondió en el puerto 11434.")
        return False
    except FileNotFoundError:
        print("❌ Ollama no está instalado o no se encontró el binario.")
        return False


def ensure_model(model_name: str = "qwen2.5:1.5b") -> None:
    """Pull the model if it isn’t already present."""
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Modelo {model_name} descargado exitosamente.")
    except subprocess.CalledProcessError:
        print(
            f"⚠️ No se pudo descargar el modelo automáticamente. "
            f"Ejecuta manualmente `ollama pull {model_name}`."
        )


if __name__ == "__main__":
    if shutil.which("ollama"):
        print("🔍 Ollama detectado en el sistema local. Iniciando servidor...")
        if start_ollama_server():
            ensure_model()
    else:
        print(
            "❗ Ollama no está disponible en este entorno. "
            "Instálalo siguiendo las instrucciones en https://ollama.com/download"
        )
