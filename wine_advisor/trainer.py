"""
Staff training material generator for the Wine Advisor.

Given a saved concept, pulls relevant wines from the database and streams
a comprehensive, print-ready training guide for front-of-house staff
via the Claude API.
"""

import os
from pathlib import Path
from typing import Generator

import anthropic

import database as db


def _make_client() -> anthropic.Anthropic:
    """
    Build an Anthropic client.
    - Prefers ANTHROPIC_API_KEY (standard setup, e.g. running on your Mac).
    - Falls back to the Claude Code session token when running inside the
      Claude Code on-web environment.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()

    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        token = Path(token_file).read_text().strip()
        return anthropic.Anthropic(auth_token=token)

    raise RuntimeError(
        "No Anthropic credentials found. Set the ANTHROPIC_API_KEY environment "
        "variable before starting the app."
    )


TRAINER_SYSTEM_PROMPT = """You are an expert wine educator and sommelier trainer with decades of \
experience developing training programs for luxury hotels, Michelin-starred restaurants, and upscale \
hospitality groups worldwide.

Your task is to generate comprehensive, practical staff training materials for a specific wine program. \
These materials will be used by front-of-house staff — servers, sommeliers-in-training, bartenders, \
and hosts — of varying wine knowledge levels.

Structure the document as follows:

# [Concept Name] — Wine Program Staff Training Guide

Open with a 2–3 paragraph introduction covering:
- The wine program's philosophy and what makes it cohesive
- How the selections reflect the concept's cuisine and guest expectations
- How staff should approach wine conversation with guests

---

Then for each wine, write a complete profile using this exact structure:

### [Vintage] [Producer] [Wine Name]
**[Appellation], [Country] | [Style] | $[Price]/bottle**
*Pronunciation: [phonetic guide for any non-English words, e.g. "Sancerre: san-SAIR"]*

**The Story**
2–3 engaging sentences a server can retell: region background, producer history, or what makes this \
wine distinctive. Focus on the human or geographical story — something memorable guests will enjoy hearing.

**In the Glass**
- **Color:** vivid descriptor (e.g. "Deep ruby with violet edges")
- **Nose:** 3–4 aromatic descriptors in plain language
- **Palate:** texture, fruit, structure
- **Finish:** length and character

**Food Pairings**
Specific dish-level suggestions matched to the concept's cuisine. Name actual dish types \
(e.g. "pan-seared duck breast", "grilled octopus with romesco") rather than broad categories. \
Include 3–4 pairing ideas.

**Selling the Wine** *(what to say at the table)*
2–3 polished, conversational sentences a server can deliver naturally when recommending this wine. \
Match the tone to the venue's positioning (casual and friendly vs. refined and authoritative).

**Service Notes**
- **Serving Temperature:** ideal range in °F/°C
- **Glassware:** specific type (e.g. "Burgundy-style bowl", "Flute or tulip")
- **Decanting:** yes/no and for how long, or "open 20 min before serving"

---

After all wine profiles, include:

## Quick Reference Table
A Markdown table with columns: Wine Name | Style | Country | Grape Varieties | ABV | Price

## Glossary
Define 10–15 wine terms used throughout the guide that staff should know, written in plain, \
approachable language. Include pronunciation for foreign terms.

---

Formatting rules:
- Use consistent Markdown heading levels throughout
- Keep language vivid and accessible — avoid impenetrable jargon
- The finished document should be ready to print or share as a PDF"""


def _format_wine(w: dict) -> str:
    """Format a wine record into a readable block for the training prompt."""
    fields = [
        f"Name: {w.get('vintage', '')} {w.get('producer', '')} {w.get('name', '')}".strip(),
        f"Style: {w.get('style', '')}",
        f"Appellation: {w.get('appellation', '')}",
        f"Region: {w.get('region', '')}",
        f"Country: {w.get('country', '')}",
        f"Grapes: {w.get('grape_varieties', '')}",
        f"Alcohol: {w.get('alcohol', '')}",
        f"Price: ${w.get('price', '?')}/{w.get('unit', 'bottle')}",
        f"Critic Score: {w.get('score', '')}",
        f"Description: {w.get('description', '')}",
    ]
    return "\n".join(f for f in fields if f.split(": ", 1)[-1].strip())


def _gather_wines(concept: dict, max_wines: int = 20) -> list[dict]:
    """Pull a representative wine set from the database for the concept."""
    seen: set[int] = set()
    wines: list[dict] = []

    def _add(results: list[dict]) -> None:
        for w in results:
            if w["id"] not in seen and len(wines) < max_wines:
                seen.add(w["id"])
                wines.append(w)

    # Search by the concept's own style/wine notes first
    if concept.get("wine_style_notes"):
        _add(db.search_wines(concept["wine_style_notes"], limit=15))

    # Then by cuisine type for pairing-relevant wines
    if concept.get("cuisine_type"):
        _add(db.search_wines(concept["cuisine_type"], limit=10))

    # Fill remaining slots with a balanced cross-style sweep
    if len(wines) < max_wines:
        for style in ["white", "red", "sparkling", "rosé", "dessert", "fortified", "orange"]:
            if len(wines) >= max_wines:
                break
            _add(db.search_wines(filters={"style": style}, limit=5))

    return wines[:max_wines]


def generate_training_stream(concept_name: str) -> Generator[str, None, None]:
    """
    Stream staff training material for a concept.
    Yields raw text strings; the caller wraps each chunk in SSE format.
    Raises ValueError if the concept is not found or the DB is empty.
    """
    concept = db.get_concept(concept_name)
    if not concept:
        raise ValueError(f"Concept '{concept_name}' not found.")

    wines = _gather_wines(concept)
    if not wines:
        raise ValueError("No wines found in the database. Please upload catalogues first.")

    wine_block = "\n\n---\n\n".join(_format_wine(w) for w in wines)

    prompt = (
        f"Please generate staff training materials for the following wine program.\n\n"
        f"**Concept:** {concept['name']}\n"
        f"**Cuisine Type:** {concept.get('cuisine_type') or 'Not specified'}\n"
        f"**Price Tier:** {concept.get('price_tier') or 'Not specified'}\n"
        f"**Guest Profile:** {concept.get('guest_profile') or 'Not specified'}\n"
        f"**Wine Style Notes:** {concept.get('wine_style_notes') or 'Not specified'}\n"
        f"**Additional Notes:** {concept.get('additional_notes') or 'None'}\n\n"
        f"**Wines in the program ({len(wines)} selected):**\n\n"
        f"{wine_block}"
    )

    client = _make_client()
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8000,
        system=TRAINER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
