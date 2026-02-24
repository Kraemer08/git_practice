# Wine Advisor — Resort Wine Buyer Assistant

A conversational AI application for luxury resort wine buyers. Upload distributor price books and catalogues, then chat with an expert sommelier AI that searches your inventory and builds curated wine lists for each restaurant concept.

## Features

- **Catalogue Ingestion** — Upload PDF or text price books; Claude (Files API) extracts every wine into a searchable SQLite database
- **Advisor Chat** — Streaming conversation with Alexandra Chen, a Master Sommelier persona powered by Claude Opus 4.6 with tool use and adaptive thinking
- **Wine Database** — Full-text search across all imported wines with filters for style, country, and price
- **Concept Profiles** — Define each restaurant/outlet (cuisine, price tier, guest profile, style preferences) and get tailored BTG + bottle suggestions
- **Placement Recommendations** — The advisor searches the live database and explains pairings, list balance, and sourcing gaps

## Architecture

```
wine_advisor/
├── app.py          Flask web app + REST API
├── advisor.py      Claude conversational AI (tool use, streaming)
├── extractor.py    Claude Files API document processing
├── database.py     SQLite with full-text search (FTS5)
└── templates/
    └── index.html  Single-page UI
```

### Claude API Features Used

| Feature | Where |
|---|---|
| **Files API** (`files-api-2025-04-14`) | Uploading and parsing PDF price books |
| **Streaming** | Both extraction and chat responses |
| **Adaptive thinking** (`claude-opus-4-6`) | Wine extraction + complex recommendations |
| **Tool use** | 6 tools: search_wines, get_wine_details, get_database_overview, save_concept, list_concepts, suggest_wines_for_concept |
| **Multi-turn conversation** | Full history sent each turn |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Run the server

```bash
cd wine_advisor
python app.py
```

Then open `http://localhost:5000` in your browser.

## Usage

### Step 1 — Upload Catalogues
Go to **Catalogues** → select a PDF or text price book → enter the supplier/importer name → click **Upload & Extract**.

Claude reads the document and extracts each wine into the database. Large catalogues (100+ pages) may take 30–60 seconds.

### Step 2 — Define Concepts
Go to **Concepts** → create a profile for each restaurant or outlet:
- The Terrace (Modern American, upscale)
- Pool Bar (casual, fruit-forward)
- Private Dining (fine dining, collector-level)
- Lobby Lounge (mid-range, approachable)

### Step 3 — Get Recommendations
Either click **Get Wine List** on a concept or go to **Advisor Chat** and ask:

> *"Build a wine list for The Terrace — 8 BTG selections, heavy on California and Burgundy, bottles up to $150 cost."*

The advisor searches the actual database, reasons about pairings, and presents a structured list with rationale.

### Chat Examples

```
What Italian whites do we carry under $30?

Compare our sparkling options for the private dining room.

We're launching a Japanese omakase pop-up. What do we have that would work, and what should I source?

Find me the best-value Pinot Noir options across all our suppliers.
```

## Supported File Types

| Type | Notes |
|---|---|
| PDF | Price books, catalogues, tasting notes |
| CSV | Tabular price lists |
| TXT | Plain text catalogues |

Excel (.xlsx) support can be added by converting to CSV before upload.
