# 🧠 RAG History Chatbot

> An intelligent Q&A chatbot powered by **Retrieval-Augmented Generation (RAG)** — ask anything from NCERT Class 10 History and get accurate, context-aware answers!

---

## 📌 What is this?

This project builds a **local RAG pipeline** that:
1. Loads the **NCERT Class 10 History PDF**
2. Splits it into semantic chunks
3. Converts chunks to **vector embeddings** using `SentenceTransformers`
4. Stores them in a **ChromaDB** vector database
5. On each query — retrieves the top-3 most relevant chunks and passes them to **LLaMA 3 (via Ollama)** for a grounded answer

No hallucinations. No internet. Just your PDF + local LLM. 🔒

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| PDF Loading | `LangChain PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `SentenceTransformers (all-MiniLM-L6-v2)` |
| Vector DB | `ChromaDB (Persistent)` |
| LLM | `LLaMA 3 8B via Ollama` |
| Orchestration | `LangChain` |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- LLaMA 3 model pulled: `ollama pull llama3:8b`

### Installation

```bash
git clone https://github.com/aksharabadal566/Rag-History-chatbot.git
cd Rag-History-chatbot
pip install -r requirement.txt
```

### Run

```bash
python app.py
```

Then type your history question when prompted!

---

## 💬 Example Usage

```
❓ Enter your question: Who was Bismarck and what role did he play in German unification?

🤖 Answer: Otto von Bismarck was the Chief Minister of Prussia who played a central role in unifying Germany. He used a combination of diplomacy and warfare — famously called "blood and iron" policy — to bring the German states together under Prussian leadership...
```

---

## 📁 Project Structure

```
Rag-History-chatbot/
├── app.py                        # Main RAG pipeline
├── NCERT-Class-10-History.pdf    # Source document
├── requirement.txt               # Dependencies
└── README.md
```

---

## 🔍 How RAG Works Here

```
User Query
    ↓
Generate Query Embedding (MiniLM)
    ↓
ChromaDB Similarity Search → Top 3 Chunks
    ↓
Build Prompt: [Context + Question]
    ↓
LLaMA 3 (Ollama) → Final Answer
```

---

## 📦 Dependencies

```
langchain
langchain-community
langchain-classic
chromadb
sentence-transformers
ollama
pypdf
```

---

## 🌟 Key Features

- ✅ **Fully local** — no API keys, no internet needed
- ✅ **Persistent vector store** — embeddings saved to disk, no re-computation
- ✅ **Context-grounded answers** — LLM only uses retrieved chunks
- ✅ **Interactive CLI** — real-time Q&A loop

---

## 🤝 Contributing

Pull requests are welcome! Feel free to extend this to other subjects or add a web UI with Flask/Streamlit.

---

## 👤 Author

**Akshara Badal**
- GitHub: [@aksharabadal566](https://github.com/aksharabadal566)
- LinkedIn: [akshara-badal566](https://linkedin.com/in/akshara-badal566)
