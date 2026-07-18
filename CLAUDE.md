# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Originally a git practice repo (the root `README.txt` and `index.html` are leftovers from that), this repository now contains **Wine Advisor** — a Flask web app for luxury resort wine buyers. Users upload distributor price books (PDF/CSV/TXT), Claude extracts each wine into a SQLite database, and a sommelier chat persona ("Alexandra Chen") answers questions and builds wine lists via tool use against that database.

All application code lives in `wine_advisor/`.

## Running the App

```bash
pip install -r wine_advisor/requirements.txt   # use this one, not the root requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
cd wine_advisor
python app.py                                   # serves http://localhost:5000 (PORT env var overrides)
```

- The root `requirements.txt` is stale (missing `pypdf`); `wine_advisor/requirements.txt` is authoritative.
- Credentials: `extractor.py:_make_client()` prefers `ANTHROPIC_API_KEY` but falls back to the Claude Code session token (`CLAUDE_SESSION_INGRESS_TOKEN_FILE`) so the app works inside Claude Code on the web. `advisor.py` currently instantiates a plain `anthropic.Anthropic()` and needs the env var.
- There are no tests, linters, or build steps in this repo.
- `sample_catalogue.csv` at the repo root is a ready-made test catalogue for the upload flow.

## Architecture

Four Python modules plus one HTML file; the data flows in two paths that meet at the database:

```
Upload path:  browser → app.py POST /api/upload → extractor.py (Claude) → database.py (SQLite)
Chat path:    browser → app.py POST /api/chat (SSE) → advisor.py agentic loop ⇄ database.py tools
```

- **`app.py`** — Flask routes (REST + one SSE endpoint). `/api/chat` streams Server-Sent Events: each event is `data: {"text": ...}` (or `{"error": ...}`), terminated by `data: [DONE]`. Uploads are saved to `wine_advisor/uploads/` (gitignored), extracted, then deleted.
- **`advisor.py`** — The conversational agent. `chat_stream(messages)` runs a while-loop: stream a Claude response (`claude-opus-4-6`), yield text tokens as they arrive, then execute any `tool_use` blocks via `_execute_tool()` and loop until the model stops calling tools. It mutates the `messages` list in place to append assistant/tool-result turns. Six tools are defined in `TOOLS`: `search_wines`, `get_wine_details`, `get_database_overview`, `save_concept`, `list_concepts`, `suggest_wines_for_concept` (the last one doesn't recommend itself — it assembles a context dump of concept profile + candidate wines by style for the model to reason over).
- **`extractor.py`** — Catalogue ingestion. PDFs are split into 50-page chunks (`_CHUNK_PAGES`) with pypdf and sent as inline base64 documents (the Files API was deliberately replaced with inline delivery — see git history); text/CSV files are embedded directly in the prompt. Claude returns a JSON array of wines parsed by `_parse_wine_json()` (tolerant of markdown fences and surrounding prose). If extraction fails, the document record is rolled back so no orphaned 0-wine entries remain.
- **`database.py`** — SQLite layer with an FTS5 virtual table (`wines_fts`) kept in sync by INSERT/DELETE triggers. `search_wines()` uses FTS `MATCH` when a text query is given, plain filtered SELECT otherwise. **`init_db()` runs on module import**, so importing `database` creates `wine_advisor/wine_advisor.db` (gitignored). The schema uses `CREATE TABLE IF NOT EXISTS`; there are no migrations — schema changes require deleting the .db file or altering it manually.
- **`templates/index.html`** — The entire single-page UI (vanilla JS, no build step): tabs for Advisor Chat, Wine Database, Catalogues, and Concepts, plus the SSE-consuming chat client.

## Conventions and Gotchas

- Claude API model ID used throughout is `claude-opus-4-6`, always via the streaming API (`client.messages.stream`) with `max_tokens=16000`.
- Wine `style` values are a fixed vocabulary (`red, white, rosé, sparkling, fortified, dessert, orange`) shared between the extractor prompt, the `search_wines` tool schema, and DB filtering — keep them in sync if changed.
- Tool results returned to Claude are plain strings (compact formatted lists or JSON dumps), not structured content blocks.
- Git history shows recurring failure modes worth remembering when touching the Claude integration: thinking-block content types breaking `b.text` access (guard with `hasattr`/block-type checks), `max_tokens` too low for extended thinking, and oversized PDFs exceeding the context window (hence the chunking).
- Development happens on `claude/...` feature branches merged into `master` via PRs.
