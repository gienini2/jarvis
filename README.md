# Jarvis — Personal AI Assistant

A self-hosted AI assistant running entirely on local hardware. No cloud LLM. Conversations via Telegram, brain on a Windows PC with an NVIDIA GPU.

## What it does

- Answers questions and holds conversations with persistent memory
- Classifies intent without calling the LLM (regex router for task commands)
- Injects only relevant context into each prompt — not a full memory dump
- Manages tasks across multiple active projects, synced from a personal dashboard
- Runs `qwen2.5:7b` locally via Ollama; falls back to Claude Haiku if Ollama is unavailable

## Architecture

```
Telegram (Hetzner VPS)
        │
        │  POST /chat  (Tailscale tunnel)
        ▼
FastAPI  :3000  (Windows PC)
        │
        ├── Intent classifier  (regex, no LLM)
        │         ├── complete_task → DB update + dashboard sync
        │         ├── query_task   → task lookup
        │         └── general      → LLM
        │
        ├── Active memory builder
        │         ├── consolidated summary (PostgreSQL)
        │         ├── relevant tasks (keyword/project match)
        │         └── recent conversation turns
        │
        └── Ollama  (qwen2.5:7b, CUDA, all layers on GPU)
                   └── fallback: Anthropic claude-haiku
```

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI (async) |
| LLM inference | Ollama — qwen2.5:7b (local) |
| GPU | NVIDIA RTX 5050, CUDA 12.0 (Blackwell) |
| Memory & tasks | PostgreSQL 16 |
| Conversation relay | python-telegram-bot on Hetzner VPS |
| Networking | Tailscale (VPS → PC tunnel) |
| Config | Pydantic BaseSettings + `.env` |

## Project structure

```
api/
  main.py          — FastAPI routes (/chat, /tasks, /health)
core/
  llm.py           — Ollama client, prompt builder
  intent.py        — Regex intent classifier + project alias map
  active_memory.py — Context injection: summaries + tasks + recent turns
  memory.py        — Conversation persistence (PostgreSQL)
  tasks.py         — Task CRUD, Centro de Mando sync
  dashboard.py     — Dashboard data loader
  summarizer.py    — Periodic conversation summarizer
  db.py            — psycopg2 connection helper
config/
  settings.py      — Pydantic settings (reads .env)
connectors/
  mando_sync.py    — Sync tasks from external dashboard JSON
scripts/
  migrate_db.py    — Database schema setup
  sync_mando.py    — One-shot task import
  summarize.py     — Manual summary trigger
  chat.py          — CLI test client
```

## Design decisions

**Intent classification before the LLM.** Task commands (`mark X as done`, `show tasks for Y`) are caught by a regex classifier and resolved directly against the database. This avoids an unnecessary LLM call for deterministic operations and keeps latency under 100ms for task queries.

**Selective context injection.** Rather than dumping all memory into the prompt, `active_memory.py` extracts keywords from the user message and fetches only matching tasks and a single consolidated summary. The prompt stays under 4K tokens while still being context-aware.

**Local-first.** The LLM runs entirely on the local GPU — no data leaves the machine. The Telegram relay is the only external component and it only forwards text; it never touches the LLM.

**Flat database schema.** Tasks, conversations, summaries, and intent logs each have their own table. No ORM — direct psycopg2 for simplicity and control.

## Setup

### Requirements

- Python 3.11+
- PostgreSQL 16 running locally
- [Ollama](https://ollama.com) with `qwen2.5:7b` pulled
- (Optional) Tailscale for the Telegram relay

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Create the database schema

```bash
python scripts/migrate_db.py
```

### 4. Pull the model

```bash
ollama pull qwen2.5:7b
```

### 5. Run

```bash
uvicorn api.main:app --port 3000
```

Test it:
```bash
curl -X POST http://localhost:3000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello"}'
```

### GPU note (NVIDIA)

Ollama offloads all model layers to GPU by default when CUDA is available. For better throughput enable Flash Attention before starting Ollama:

**Windows:**
```
setx OLLAMA_FLASH_ATTENTION 1
```
Then restart Ollama from the system tray.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Main chat endpoint |
| `GET` | `/tasks` | List tasks (filter by `project`, `keyword`, `status`) |
| `POST` | `/tasks/sync` | Re-import tasks from dashboard JSON |
| `POST` | `/tasks/context` | Attach a note/link/decision to a task |
| `GET` | `/health` | Liveness check |

## Environment variables

See `.env.example`. The only required variables are the PostgreSQL connection details. Everything else has a default.
