# Multimodal AI Chat

New AI-assisted portfolio reconstruction (November 2026), not the original resume project. Streamlit interface for local text chat (Mistral), image understanding (LLaVA), audio transcription (faster-whisper base, CPU int8), and page-grounded PDF RAG (ChromaDB and Ollama embeddings). Model responses and citations are fallible; review the displayed retrieved evidence. No clinical, confidential, or production data is included.

## Installation
Python 3.11 recommended. Install [Ollama](https://ollama.com/) and keep its server running. Pull the models:

```bash
ollama pull mistral
ollama pull llava
ollama pull nomic-embed-text
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```
Whisper downloads the base model on first use. Model downloads require internet and several GB of disk; memory and latency depend on hardware. Use the sidebar to switch tasks. For PDF QA, upload a text-based PDF, click Index PDF, then ask a question. The Chroma collection is ephemeral, and retrieval uses chunk embeddings. Scanned PDFs require separate OCR.

## Test
`python -m pytest -q` tests chunking, prompt generation and mocked retrieval. Full local model, Streamlit and Chroma integration require the installed models/dependencies and have not been verified in the build environment. Quantization and performance vary by downloaded model; no benchmark is claimed.
