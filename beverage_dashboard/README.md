# Beverage Program Dashboard

Upload your daily POS reports and get an instant read on the beverage program at
each outlet — where the revenue is, how drinks attach to covers, which items are
stars vs. dead weight, and an AI briefing on **where the biggest opportunities
are**. Built for the Regent Santa Monica Beach outlets (Orla today; Azure, IRD
and Banquets are wired in and ready for data).

## What it does

- **Ingests InfoGenesis POS exports** — drop in a *Product Mix-Revenue* report
  (item detail) and/or a *Sales Summary with Revenue Detail* report (covers,
  checks, average check). PDF, CSV or TXT.
- **Classifies every line** with Claude into beverage sub-types — cocktail,
  spirit, wine by the glass, wine by the bottle, beer, N/A beverage, coffee &
  tea — the nuance raw POS category codes miss (a "Liquor" line might be a
  signature cocktail *or* a spirit pour; coffee and soda hide under "Food").
- **Dashboards the numbers** — KPIs (beverage revenue, % of total, **beverage &
  alcohol per cover**), category and beverage mix, day-over-day trend, top
  items, and a **Volume × Price opportunity map** (Stars / Volume Drivers /
  Premium / Sleepers).
- **AI opportunity briefing** — Claude reads the aggregated numbers and briefs
  you like a beverage director: what to push, reprice, feature, or cut, with the
  supporting figures.

## Architecture

```
beverage_dashboard/
├── app.py         Flask web app + REST API
├── ingest.py      Claude-powered report parsing + beverage classification
├── analytics.py   KPIs, mix, trend, opportunity matrix, AI briefing
├── database.py    SQLite storage (reports, sales_items, outlet_days)
└── templates/
    └── index.html Single-page dashboard (vanilla JS + inline SVG charts)
```

### Claude features used

| Feature | Where |
|---|---|
| **Forced tool use** (structured output) | Extracting line items & summary totals |
| **Classification** | Tagging every item with a beverage sub-type |
| **Analytical reasoning** | The "where's the opportunity" briefing |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # or run inside Claude Code on the web
cd beverage_dashboard
python app.py                            # http://localhost:5001
```

## Using it

1. Click **Upload Report**, pick the outlet, and drop in a daily *Sales Summary*
   (for covers/checks) and *Product Mix-Revenue* (for item detail). Upload as
   many days as you like — trends and momentum unlock at two-plus days.
2. Read the dashboard: KPIs up top, mix in the middle, the **Opportunity Map**
   and quadrant lists below.
3. Hit **Generate briefing** for the AI read on where to steer the program.

Re-uploading a corrected report for the same outlet/date replaces the old one,
so numbers never double-count.

## A note on margin

These POS reports carry **revenue but no product cost**, so every "opportunity"
here is framed on revenue, mix share, price point and attachment-per-cover — not
profit. Add an item cost / pour-cost column later and this unlocks true
profit-based menu engineering (real Stars/Plowhorses/Puzzles/Dogs).

## Supported reports

| Report | Gives us |
|---|---|
| Product Mix-Revenue | Item-level units, gross/net revenue, discounts, category |
| Sales Summary with Revenue Detail | Covers, checks, average check, category totals |

Both are standard InfoGenesis POS exports. Other POS formats (Toast, Micros,
Simphony) generally export the same shape and can be added the same way.
