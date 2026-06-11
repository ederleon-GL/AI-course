import json, pathlib, sys

# Path to the notebook that contains the Colab‑specific cell
NOTEBOOK_PATH = pathlib.Path('/Users/guane/Documentos/GlogalLogic/AI-course/notebooks/01_introduction/04_intro_langchain.ipynb')

# Load notebook JSON
with NOTEBOOK_PATH.open('r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False

# Keywords that indicate Colab‑specific content (case‑insensitive)
CODE_KEYWORDS = ['google.colab', 'Detectar si estamos en Google Colab']
MARKDOWN_KEYWORDS = [
    'colab',
    'curl -fs',
    'instalando ollama en google colab',
    'instalación en colab',
    'google colab',
    'ollama install',
]

# Full Ollama‑local setup code to embed directly
OLLAMA_SETUP_CODE = [
    "import os, subprocess, time, shutil, socket\n",
    "\n",
    "def start_ollama_server():\n",
    "    try:\n",
    "        subprocess.Popen([\"ollama\", \"serve\"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n",
    "        for _ in range(30):\n",
    "            try:\n",
    "                s = socket.socket()\n",
    "                s.settimeout(1)\n",
    "                s.connect((\"127.0.0.1\", 11434))\n",
    "                s.close()\n",
    "                print(\"✅ Servidor Ollama listo en el puerto 11434!\")\n",
    "                return True\n",
    "            except Exception:\n",
    "                time.sleep(1)\n",
    "        print(\"❌ Error: el servidor Ollama no respondió en el puerto 11434.\")\n",
    "        return False\n",
    "    except FileNotFoundError:\n",
    "        print(\"❌ Ollama no está instalado o no se encontró el binario.\")\n",
    "        return False\n",
    "\n",
    "if shutil.which(\"ollama\"):\n",
    "    print(\"🔍 Ollama detectado en el sistema local. Iniciando servidor...\")\n",
    "    if start_ollama_server():\n",
    "        try:\n",
    "            subprocess.run([\"ollama\", \"pull\", \"qwen2.5:1.5b\"], check=True)\n",
    "            print(\"✅ Modelo qwen2.5:1.5b descargado exitosamente.\")\n",
    "        except subprocess.CalledProcessError:\n",
    "            print(\"⚠️ No se pudo descargar el modelo automáticamente. Por favor, instala el modelo manualmente con `ollama pull qwen2.5:1.5b`.\")\n",
    "else:\n",
    "    print(\"❗ Ollama no está disponible en este entorno. Instálalo siguiendo las instrucciones en https://ollama.com/download\")\n",
]

for cell in nb.get('cells', []):
    # ---------------------------------------------------------------
    # 1️⃣ Clean code cells that reference Colab
    # ---------------------------------------------------------------
    if cell.get('cell_type') == 'code':
        source_text = ''.join(cell.get('source', []))
        if any(k in source_text for k in CODE_KEYWORDS):
            # Replace the whole cell with the embedded Ollama local setup code
            cell['source'] = OLLAMA_SETUP_CODE
            changed = True
            continue

    # ---------------------------------------------------------------
    # 2️⃣ Clean markdown cells: drop lines mentioning Colab or old install flow
    #    and adapt comments to mention a local‑only environment.
    # ---------------------------------------------------------------
    if cell.get('cell_type') == 'markdown':
        new_source = []
        for line in cell.get('source', []):
            lowered = line.lower()
            if any(keyword in lowered for keyword in MARKDOWN_KEYWORDS):
                continue
            adapted_line = line.replace('Google Colab', 'tu máquina local')
            adapted_line = adapted_line.replace('Colab', 'entorno local')
            new_source.append(adapted_line)
        if new_source != cell.get('source', []):
            cell['source'] = new_source
            changed = True

# ---------------------------------------------------------------
# 3️⃣ Optionally remove empty cells (no source after cleaning)
# ---------------------------------------------------------------
cleaned_cells = []
for cell in nb.get('cells', []):
    if cell.get('source'):
        cleaned_cells.append(cell)
    else:
        changed = True
nb['cells'] = cleaned_cells

if changed:
    with NOTEBOOK_PATH.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook updated: Colab code replaced with embedded Ollama local setup.')
else:
    print('No Colab‑specific content found; no changes made.')
