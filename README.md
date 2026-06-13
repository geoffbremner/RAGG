# RAGG - Retrieval Augmented Geoff Generation

MANUALLY WRITTEN SECTION

This is a working V1 of - Retrieval Augmented Geoff Generation - developed by Geoff Bremner: https://linktr.ee/gbaudio

Usage: 

Use the python terminal interface, copy paste RAG prompts into any LLM! :)

Work with ANY corpus of fact context you have in docs/ directory. This will make FACT BASED data in your prompts.

# 🚀 Private Local Python RAG System & Prompt Ingestion Sandbox

A lightweight, high-performance, and fully local Retrieval-Augmented Generation (RAG) pipeline written in pure Python. Drop your unstructured document formats (PDFs, Markdown, text files, code) into a local staging directory to automatically segment them, index them in a local persistent vector database, and perform context-stuffed, copy-paste ready prompting bounded strictly to your factual source boundaries.

---

## ✨ System Architecture & Key Features

* **Strict Boundary Grounding (Zero Hallucination)**: Compiles matching segments beside customized instructions to form a cohesive context-stuffed prompt. Perfect for drop-in use with ChatGPT, Claude, Gemini, or any secondary model.
* **Smart Semantic Paragraph Chunking**: Unlike character-based splitters that severed concepts, the pipeline tokenizes texts by logical boundaries (e.g. paragraphs and double line breaks) first, preserving cohesive thought models.
* **Context Preservation (Sliding Overlap Window)**: Integrates slide boundary tracking (`chunk_size` and `chunk_overlap`) to ensure keywords or transitions don't get truncated at chunk borders.
* **Hybrid Search Retrieval**: Supports 100% offline, pure-Python cosine-similarity and term vector models, or upgrades instantly to high-density vector indices (Chroma SQLite standard DB) when backed by live Gemini or OpenAI APIs!
* **Configurable Subject Matter Persona Routing**: Firing up the command line with custom subject tags dynamically adapts instructions, shaping expert domain roles (e.g. Legal Auditor, Technical Architect, Medical Analyst).
* **Live In-Console Inference**: Detected credentials present live, streaming generation results right back into your active terminal shell session!

---

## ⚡ Prerequisites

This repository runs locally on **macOS**, or **Linux** systems.

### Runtime Engines
Verify that your system hosts current installations of:
* **Python**: `3.9` to `3.11` is recommended.

```bash
python3 --version
```

If you do not have Python installed, obtain it instantly using the [Homebrew package manager](https://brew.sh) (on Mac):
```bash
brew install python
```

---

## 🚀 Step-by-Step Local Setup

Isolate your workspace namespace securely and begin querying:

### 1. Create and Activate the Virtual Environment
Create a lightweight, isolated Python virtual sandbox strictly inside this workspace:
```bash
# Generate the virtual environment container folder
python3 -m venv venv

# Activate active sandbox session configurations
# On macOS and Linux:
source venv/bin/activate

# On Windows (Command Prompt):
# .\venv\Scripts\activate
```
*(Once activated, your terminal shell prefix will display `(venv)` to indicate strict dependency isolation).*

### 2. Install Project Dependencies
Install standard vector embedding, document parsing, and model orchestration modules directly into your activated environment wrapper:
```bash
pip install -r requirements.txt
```

### 3. Load Your Datasets / Documents folder
Create the `docs/` staging folder locally and copy your knowledge documents (unstructured text `.txt`, markdown `.md`, or `.pdf` file types) straight inside:
```bash
mkdir -p docs
# Drop your knowledge materials, logs, or PDFs inside the ./docs/ directory!
```

---

## 🐳 Running the Program (Dynamic CLI CLI Mode)

Open your active terminal session and launch the program:

```bash
python3 rag_pipeline.py
```

### Dynamic Flags
Optimize search scope variables during boot using the built-in argument parser flags:

* **Specify Domain Subject Context Role (`--subject`)**:
  ```bash
  python3 rag_pipeline.py --subject "YOUR WORKING SUBJECT"
  ```
* **Adjust Context Retrieval Depth Count (`--k`)**:
  ```bash
  python3 rag_pipeline.py --k 5
  ```

---

## 📁 Key Directories & Persistent State Files

* `/docs/` — Directory containing unstructured seed documents. Drop any number of txt, md, or pdf files in here.
* `/chroma_db/` — Created automatically during initialization. Refers to the persistent SQLite-backed vector search directory.
* `/rag_pipeline.py` — The core RAG script containing the main loop, split segmenters, index engines, and terminal builders.
* `/requirements.txt` — Manifest containing required Python modules.

### Reset Database Cache
To wipe the indexed documents vectors and re-index your `/docs/` contents from scratch, simply delete the local database directory:
`rm -rf chroma_db/`