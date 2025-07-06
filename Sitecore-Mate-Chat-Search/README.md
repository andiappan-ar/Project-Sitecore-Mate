# Mate - Sitecore Chat Search

> 🧑‍💻 Some days, AI is the teacher; other days, the eager student.\
> 🤖 But these days, AI’s my mate.

## Overview

**Sitecore Chat Search** is a solution designed to scrape, index, and query content from Sitecore instances, enabling semantic search and Retrieval-Augmented Generation (RAG) capabilities.\
It features a **Next.js** frontend for environment management and search, and a **Python FastAPI** backend for content processing, embeddings, and LLM interaction.

**Key features include:**

- 🧩 **Advanced Content Processing:** Scrapes page-level and component-level fields, with HTML cleaning for high-quality text.
- 📏 **Intelligent Chunking:** Recursively splits content into optimal chunks (100–300 tokens, 10–20 overlap) for context-preserving embeddings.
- 📦 **Vector Database Integration:** Uses ChromaDB for efficient semantic search.
- 🤹 **Flexible LLM Support:** Works with Gemini, OpenAI, and Ollama models for RAG-based QA and summarization.
- 🚀 **Advanced RAG (Optional):** Supports LLM-generated summaries of pages and components for richer retrieval.
- 📡 **Real-time Logging:** Live scraping and indexing logs from backend to frontend.
- 🌍 **Multilingual Support:** Handles multiple languages for international content.

---

## 🏗️ Architecture

The solution is split into two main components:

### 1. ⚛️ Next.js Frontend (`Sitecore-Content-Scrapper-Website`)

- User-friendly UI for:
  - 🛠️ Configuring Sitecore environments
  - 🗂️ Scraping/indexing content
  - 🖥️ Viewing logs in real-time
  - 🔎 Vector searches and conversational RAG
- Acts as a proxy to the Python backend

### 2. 🐍 Python FastAPI Backend (`Sitecore-Chat-Search-MCP`)

- Handles:
  - 🗃️ Connection to ChromaDB (`./chroma_db`)
  - 🧬 Embedding with `SentenceTransformer` (`all-MiniLM-L6-v2`)
  - 🧹 HTML processing, chunking, and validation
  - 📊 Vector DB management (add/query)
  - 🧠 LLM interaction for summarization and QA
  - 🔗 Streaming logs to frontend via SSE

---

## 📂 Project Structure (High-Level)

```
Sitecore-Mate-Chat-Search/
├── Sitecore-Chat-Search-MCP/           # Python FastAPI Backend
│   ├── main.py                         # FastAPI app, endpoints
│   ├── llm_config.py                   # LLM config and clients
│   ├── prompts.py                      # Prompt templates
│   ├── requirements.txt
│   ├── sitecore_chunking_vectordb_plan.md
│   └── .env
└── Sitecore-Content-Scrapper-Website/  # Next.js Frontend
    ├── src/
    │   ├── app/
    │   │   ├── api/
    │   │   └── ...
    │   ├── graphql/
    │   ├── index/
    │   └── lib/
    ├── environments.json
    ├── next.config.ts
    ├── package.json
    └── .env.local
```

---

## 🧰 Technologies Used

**Frontend:**

- ⚛️ Next.js 15.3.4
- ⚛️ React 19.0.0
- 🎨 Tailwind CSS 4.1.11, PostCSS 8.5.6
- 🔗 GraphQL-Request 7.2.0
- 📝 Fast-XML-Parser 5.2.5
- 🆔 UUID 4.x

**Backend:**

- 🐍 FastAPI
- 📝 Pydantic
- 📦 ChromaDB 3.0.6
- 🧬 Sentence Transformers (`all-MiniLM-L6-v2`)
- 🍲 BeautifulSoup4
- 🪢 Langchain
- 🤖 Google Generative AI SDK
- 🤖 OpenAI SDK
- 🤖 Ollama

---

## 🚦 Getting Started

### Prerequisites

- 🟢 Node.js (v18+)
- 🟢 Python (v3.9+)
- 🟢 npm/yarn/pnpm/bun
- 🟢 (Optional) Ollama desktop with models pulled

### 1. Backend Setup

```bash
cd Sitecore-Mate-Chat-Search/Sitecore-Chat-Search-MCP
python -m venv venv
# Activate venv, then:
pip install -r requirements.txt
# Or install individually as needed
```

- Configure `.env` (see example above for model/API options)
- Run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 2. Frontend Setup

```bash
cd Sitecore-Mate-Chat-Search/Sitecore-Content-Scrapper-Website
npm install
# Configure .env.local with NEXT_PUBLIC_PYTHON_BASE_API_URL etc.
npm run dev
```

The app runs at [http://localhost:3000](http://localhost:3000).

---

## 🏁 Usage

1. **Access**: Visit `http://localhost:3000`
2. **Manage Environments**: Add and configure Sitecore environments.
3. **Scrape & Index**: Use the UI to trigger scraping and see real-time logs.
4. **AI Content Search**: Query via semantic search or RAG.

---

## 🛠️ Advanced Configuration

- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `N_RESULTS`, `ADVANCED_RAG_ENABLED`, and LLM provider options are in your backend `.env`.
- Frontend `.env.local` controls API URL and defaults.

---

## 🚀 Deployment

- **Frontend**: Deploy easily on [Vercel](https://vercel.com/) or see [Next.js deployment docs](https://nextjs.org/docs/app/building-your-application/deploying).
- **Backend**: Deploy with Gunicorn, or on cloud services like GCP, AWS, or Azure.

---

## 🤝 Contribution

Contributions welcome—feel free to open issues or PRs!

## 📜 License

MIT License

---

> 🚧 This project is a work in progress—check back for updates and new “mates” joining the Sitecore journey!

