"""
Wine Advisor: a conversational AI layer powered by Claude with tool use.

The advisor acts as a deeply knowledgeable wine buyer / sommelier.
It has access to tools that search the local wine database and manage
restaurant concept profiles, then synthesises placement recommendations.
"""

import json
from typing import Generator

import anthropic

import database as db

client = anthropic.Anthropic()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Alexandra Chen, a Master Sommelier and senior wine buyer for a collection
of luxury restaurants at a prestigious resort in Los Angeles. You have access to a live database of
wines from the resort's approved importers and distributors.

Your expertise spans:
- All major and obscure wine regions globally
- Food and wine pairing across diverse cuisines
- Cellar management, vintages, and aging potential
- Luxury market positioning and guest experience
- Supplier relationships and negotiating value

Your role is to help the resort's wine buyer by:
1. Answering questions about wines in the database using search_wines and get_wine_details
2. Helping define restaurant concept profiles using save_concept and list_concepts
3. Generating detailed, reasoned wine list suggestions using suggest_wines_for_concept
4. Discussing wine trends, regions, producers, and strategy

When making placement suggestions:
- Tailor selections to each concept's cuisine, price tier, and guest profile
- Suggest BTG (by the glass) and bottle options with reasoning
- Note pairing rationale, seasonal considerations, and list balance (reds/whites/sparkling)
- Flag any gaps in the current database that should be sourced

Be direct, authoritative, and specific. Use proper wine nomenclature. Reference actual wines
from the database by name and producer. When you don't find a good match, say so clearly
and suggest what styles should be sought from suppliers.

Always think step-by-step when building a wine list recommendation."""


# ── Tool definitions ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_wines",
        "description": (
            "Search the wine database by any combination of free-text query and structured filters. "
            "Use this to find wines by name, producer, region, grape variety, style, or description. "
            "Returns up to 50 matching wines with full details."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search across name, producer, region, grape, description. Leave empty to browse all.",
                },
                "style": {
                    "type": "string",
                    "enum": ["red", "white", "rosé", "sparkling", "fortified", "dessert", "orange"],
                    "description": "Filter by wine style.",
                },
                "country": {
                    "type": "string",
                    "description": "Filter by country of origin, e.g. 'France', 'Italy'.",
                },
                "min_price": {
                    "type": "number",
                    "description": "Minimum bottle price (USD).",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum bottle price (USD).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_wine_details",
        "description": "Retrieve complete details for a specific wine by its database ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wine_id": {
                    "type": "integer",
                    "description": "The wine's database ID (from search results).",
                },
            },
            "required": ["wine_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_database_overview",
        "description": (
            "Get a summary of the wine database: total count, countries represented, "
            "styles available, and imported suppliers. Useful before making recommendations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "save_concept",
        "description": (
            "Save or update a restaurant or outlet profile. "
            "Use this when the buyer describes a concept so you can tailor recommendations to it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Concept name, e.g. 'The Terrace Restaurant', 'Pool Bar', 'Private Dining'.",
                },
                "cuisine_type": {
                    "type": "string",
                    "description": "Cuisine style, e.g. 'Modern American', 'Japanese omakase', 'Mediterranean'.",
                },
                "price_tier": {
                    "type": "string",
                    "enum": ["casual", "mid-range", "upscale", "fine dining", "ultra-premium"],
                    "description": "Pricing tier that describes the outlet's positioning.",
                },
                "guest_profile": {
                    "type": "string",
                    "description": "Description of typical guests, e.g. 'Resort leisure guests, mostly couples, prefer accessible wines'.",
                },
                "wine_style_notes": {
                    "type": "string",
                    "description": "Wine style preferences or restrictions, e.g. 'Heavy on natural and biodynamic; avoid heavy oaky reds'.",
                },
                "additional_notes": {
                    "type": "string",
                    "description": "Any other relevant context: glass program size, cellar budget, seasonal focus, etc.",
                },
            },
            "required": ["name", "cuisine_type", "price_tier"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_concepts",
        "description": "List all saved restaurant and outlet concepts.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "suggest_wines_for_concept",
        "description": (
            "Generate a detailed, curated wine list suggestion for a named concept. "
            "Searches the database for suitable wines and produces BTG and bottle recommendations "
            "with pairing rationale. Provide specific requirements if desired."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concept_name": {
                    "type": "string",
                    "description": "Name of the saved concept to build the list for.",
                },
                "requirements": {
                    "type": "string",
                    "description": "Additional buyer requirements for this list, e.g. 'Focus on Italian; 6 BTG whites, 4 BTG reds; under $25/bottle cost'.",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum wholesale bottle price to include.",
                },
            },
            "required": ["concept_name"],
            "additionalProperties": False,
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict) -> str:
    if name == "search_wines":
        filters = {k: inputs[k] for k in ("style", "country", "min_price", "max_price") if k in inputs}
        wines = db.search_wines(query=inputs.get("query", ""), filters=filters)
        if not wines:
            return "No wines found matching those criteria."
        summary = []
        for w in wines:
            summary.append(
                f"[ID {w['id']}] {w.get('vintage','')} {w.get('producer','')} {w.get('name','')} "
                f"({w.get('style','')}, {w.get('region','')}, {w.get('country','')}) "
                f"${w.get('price','?')}/{w.get('unit','bottle')} | {w.get('grape_varieties','')}"
            )
        return f"Found {len(wines)} wines:\n" + "\n".join(summary)

    elif name == "get_wine_details":
        wine = db.get_wine(inputs["wine_id"])
        if not wine:
            return f"No wine found with ID {inputs['wine_id']}."
        return json.dumps(wine, indent=2)

    elif name == "get_database_overview":
        stats = db.get_wine_stats()
        return json.dumps(stats, indent=2)

    elif name == "save_concept":
        db.upsert_concept(
            name=inputs["name"],
            cuisine_type=inputs.get("cuisine_type", ""),
            price_tier=inputs.get("price_tier", ""),
            guest_profile=inputs.get("guest_profile", ""),
            wine_style_notes=inputs.get("wine_style_notes", ""),
            additional_notes=inputs.get("additional_notes", ""),
        )
        return f"Concept '{inputs['name']}' saved successfully."

    elif name == "list_concepts":
        concepts = db.list_concepts()
        if not concepts:
            return "No concepts saved yet."
        lines = []
        for c in concepts:
            lines.append(
                f"• {c['name']} — {c['cuisine_type']} | {c['price_tier']}\n"
                f"  Guests: {c.get('guest_profile','')}\n"
                f"  Wine notes: {c.get('wine_style_notes','')}"
            )
        return "\n\n".join(lines)

    elif name == "suggest_wines_for_concept":
        concept = db.get_concept(inputs["concept_name"])
        if not concept:
            return f"Concept '{inputs['concept_name']}' not found. Save it first with save_concept."

        max_price = inputs.get("max_price")
        requirements = inputs.get("requirements", "")

        # Gather wines to work with
        wine_lists = {}
        for style in ["white", "red", "sparkling", "rosé", "dessert", "fortified"]:
            filters: dict = {"style": style}
            if max_price:
                filters["max_price"] = max_price
            wines = db.search_wines(filters=filters, limit=30)
            if wines:
                wine_lists[style] = wines

        if not wine_lists:
            return "The wine database is empty. Please upload catalogues first."

        context = f"""Concept Profile:
Name: {concept['name']}
Cuisine: {concept['cuisine_type']}
Price tier: {concept['price_tier']}
Guest profile: {concept.get('guest_profile','')}
Wine style notes: {concept.get('wine_style_notes','')}
Additional notes: {concept.get('additional_notes','')}

Buyer requirements: {requirements or 'None specified'}

Available wines by style:
"""
        for style, wines in wine_lists.items():
            context += f"\n{style.upper()} ({len(wines)} options):\n"
            for w in wines[:15]:
                context += (
                    f"  [ID {w['id']}] {w.get('vintage','')} {w.get('producer','')} "
                    f"{w.get('name','')} ({w.get('appellation') or w.get('region','')}, "
                    f"{w.get('country','')}) ${w.get('price','?')}/{w.get('unit','bottle')} "
                    f"| {w.get('grape_varieties','')}\n"
                )

        return context

    return f"Unknown tool: {name}"


# ── Main advisor loop ──────────────────────────────────────────────────────────

def chat_stream(messages: list[dict]) -> Generator[str, None, None]:
    """
    Run one turn of the agentic loop with real-time streaming text output.
    Yields text chunks as they arrive from Claude; handles tool calls internally.
    messages: full conversation history (user + assistant turns).
    """
    while True:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 8000},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            # Yield text tokens in real time as they arrive
            for text in stream.text_stream:
                yield text
            response = stream.get_final_message()

        # Collect tool calls from the completed response
        tool_calls = [b for b in response.content if b.type == "tool_use"]

        # If no tool calls, we're done
        if not tool_calls:
            break

        # Execute tools and loop back for the next assistant turn
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tc in tool_calls:
            result = _execute_tool(tc.name, tc.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})
