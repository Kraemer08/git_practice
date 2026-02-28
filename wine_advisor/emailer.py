"""
Email drafting module for supplier communications.

- draft_tasting_request(): generates an outbound tasting appointment email
  tailored to a supplier's portfolio and a specific restaurant concept.
- process_inbound_email(): classifies an inbound supplier email and drafts
  an appropriate professional response.

Both functions use tool_use for structured, reliable output, and personalize
emails using the buyer's saved profile (name, title, company, writing style).
"""

import os
from pathlib import Path

import anthropic

import database as db


def _make_client() -> anthropic.Anthropic:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return anthropic.Anthropic()
    token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE")
    if token_file and Path(token_file).exists():
        token = Path(token_file).read_text().strip()
        return anthropic.Anthropic(auth_token=token)
    raise RuntimeError("No Anthropic credentials found.")


client = _make_client()


# ── Profile helpers ────────────────────────────────────────────────────────────

def _buyer_signature(profile: dict) -> str:
    """Build the email sign-off block from the buyer's profile."""
    name     = (profile.get("name")     or "").strip()
    title    = (profile.get("title")    or "").strip()
    company  = (profile.get("company")  or "").strip()
    location = (profile.get("location") or "").strip()

    lines = []
    if name:
        lines.append(name)
    role_line = title
    if company:
        role_line += (" | " if role_line else "") + company
    if location:
        role_line += (", " if role_line else "") + location
    if role_line:
        lines.append(role_line)

    return "\n".join(lines) if lines else "Wine Buyer"


def _buyer_identity(profile: dict) -> str:
    """One-line identity string for system prompt preamble."""
    name     = (profile.get("name")     or "the wine buyer").strip()
    title    = (profile.get("title")    or "").strip()
    company  = (profile.get("company")  or "").strip()
    location = (profile.get("location") or "").strip()

    identity = name
    if title:
        identity += f", {title}"
    if company and location:
        identity += f" at {company} in {location}"
    elif company:
        identity += f" at {company}"
    return identity


def _build_outbound_system(profile: dict) -> str:
    identity      = _buyer_identity(profile)
    style_summary = (profile.get("style_summary") or "").strip()

    system = (
        f"You are {identity}. You are writing a professional email to a wine supplier "
        f"requesting a tasting appointment.\n\n"
        "These are collegial ongoing relationships — your tone is friendly and direct, "
        "never generic. You always reference specific wines from their portfolio and clearly "
        "explain why each one interests you. Keep emails concise: a brief intro, the wines "
        "you want to taste, and a flexible scheduling ask (under 300 words)."
    )
    if style_summary:
        system += f"\n\nYour personal writing style — match this voice exactly:\n{style_summary}"
    return system


def _build_inbound_system(profile: dict) -> str:
    identity      = _buyer_identity(profile)
    style_summary = (profile.get("style_summary") or "").strip()

    system = (
        f"You are {identity}. A wine supplier or importer rep has sent you an email. "
        "Read it, classify it, and draft a professional response.\n\n"
        "Email types:\n"
        "- tasting_request   — Rep asking to schedule a tasting with you\n"
        "- buyer_lunch       — Invitation to a winemaker dinner, trade lunch, or similar\n"
        "- allocation_offer  — Offer of allocated, futures, or limited-availability wines\n"
        "- vintage_release   — New vintage or portfolio release announcement\n"
        "- pricing_update    — Updated price list or cost changes\n"
        "- invoice_follow_up — Payment, invoice, or account query\n"
        "- general_inquiry   — General check-in, follow-up, or newsletter\n"
        "- other             — Anything else\n\n"
        "Response tone: professional, warm, and specific. Reference details from their "
        "email. Keep replies concise and clear about next steps."
    )
    if style_summary:
        system += f"\n\nYour personal writing style — match this voice exactly:\n{style_summary}"
    return system


# ── Outbound: tasting request ──────────────────────────────────────────────────

_OUTBOUND_TOOLS = [
    {
        "name": "create_email_draft",
        "description": "Output the completed tasting request email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line",
                },
                "body": {
                    "type": "string",
                    "description": "Full email body as plain text (no markdown). Use blank lines between paragraphs.",
                },
                "wines_highlighted": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Names of the 4–8 wines selected to highlight",
                },
            },
            "required": ["subject", "body"],
        },
    }
]


def draft_tasting_request(supplier_doc_id: int, concept_name: str,
                          notes: str = "") -> dict:
    """
    Generate a tasting request email for a supplier, tailored to a concept.
    Returns dict: {subject, body, wines_highlighted}.
    """
    profile = db.get_user_profile()

    docs = db.list_documents()
    doc = next((d for d in docs if d["id"] == supplier_doc_id), None)
    if not doc:
        raise ValueError(f"Supplier document {supplier_doc_id} not found.")

    supplier_name        = doc.get("supplier") or doc["filename"]
    contact_name         = doc.get("contact_name") or ""
    contact_email        = doc.get("contact_email") or ""
    relationship_notes   = (doc.get("relationship_notes") or "").strip()

    concept = db.get_concept(concept_name) if concept_name else None

    wines = db.search_wines(filters={"doc_id": supplier_doc_id}, limit=60)
    if not wines:
        raise ValueError(
            f"No wines found for '{supplier_name}'. Make sure their catalogue is uploaded."
        )

    wine_lines = "\n".join(
        f"  • {w.get('vintage', '')} {w.get('producer', '')} {w.get('name', '')} "
        f"({w.get('style', '')}, {(w.get('appellation') or w.get('region') or '').strip()}, "
        f"{w.get('country', '')}) — ${w.get('price', '?')}/{w.get('unit', 'bottle')}"
        for w in wines
    )

    concept_block = ""
    if concept:
        concept_block = f"""
Restaurant concept I am building for:
  Name:        {concept['name']}
  Cuisine:     {concept['cuisine_type']}
  Price tier:  {concept['price_tier']}
  Guests:      {concept.get('guest_profile', '')}
  Wine notes:  {concept.get('wine_style_notes', '')}
"""

    relationship_block = ""
    if relationship_notes:
        relationship_block = f"\nContext about this rep relationship: {relationship_notes}\n"

    greeting     = contact_name if contact_name else "the sales team"
    contact_line = f" ({contact_email})" if contact_email else ""

    prompt = f"""Draft a tasting request email to {supplier_name}, addressed to {greeting}{contact_line}.
{concept_block}{relationship_block}
Additional buyer notes: {notes or 'None.'}

Full wine portfolio from this supplier:
{wine_lines}

Select 4–8 wines that best fit the concept and notes. For each chosen wine, briefly note why it's interesting (style fit, price point, uniqueness, etc.). Keep the email under 300 words.

Sign off as:
{_buyer_signature(profile)}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=_build_outbound_system(profile),
        tools=_OUTBOUND_TOOLS,
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "create_email_draft":
            return dict(block.input)

    raise RuntimeError("Email draft generation failed — no tool call in response.")


# ── Inbound: classify and respond ─────────────────────────────────────────────

_INBOUND_TOOLS = [
    {
        "name": "draft_response",
        "description": "Classify the email and output a drafted reply.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email_type": {
                    "type": "string",
                    "enum": [
                        "tasting_request", "buyer_lunch", "allocation_offer",
                        "vintage_release", "pricing_update", "invoice_follow_up",
                        "general_inquiry", "other",
                    ],
                    "description": "Classified type of the inbound email",
                },
                "email_type_label": {
                    "type": "string",
                    "description": "Human-readable label, e.g. 'Tasting Request'",
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line for your reply",
                },
                "body": {
                    "type": "string",
                    "description": "Full reply body as plain text. Use blank lines between paragraphs.",
                },
                "action_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key actions or decisions flagged by this email (may be empty list)",
                },
            },
            "required": ["email_type", "email_type_label", "subject", "body"],
        },
    }
]


def process_inbound_email(email_text: str) -> dict:
    """
    Classify an inbound supplier email and draft a response.
    Returns dict: {email_type, email_type_label, subject, body, action_items}.
    """
    if not email_text.strip():
        raise ValueError("Email text is empty.")

    profile = db.get_user_profile()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=_build_inbound_system(profile),
        tools=_INBOUND_TOOLS,
        tool_choice={"type": "any"},
        messages=[{
            "role": "user",
            "content": (
                f"Here is the supplier email:\n\n---\n{email_text}\n---\n\n"
                f"Sign your response as:\n{_buyer_signature(profile)}"
            ),
        }],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "draft_response":
            return dict(block.input)

    raise RuntimeError("Email processing failed — no tool call in response.")
