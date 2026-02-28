"""
Writing style analysis for personalizing email drafts to match the buyer's voice.
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


def analyze_writing_style(samples: str) -> str:
    """
    Analyze email writing samples and return a concise style guide (4-6 bullets).
    Saves the result to the user profile in the database.
    Returns the style summary string.
    """
    if not samples.strip():
        raise ValueError("No writing samples provided.")

    client = _make_client()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                "Analyze these email writing samples from a wine buyer and produce a concise "
                "style guide (4–6 bullet points) that captures their authentic voice.\n\n"
                "Focus on:\n"
                "- Tone and formality level (e.g. warm/casual, direct/formal)\n"
                "- How they open and close emails\n"
                "- Vocabulary patterns (wine terminology use, sentence length, word choice)\n"
                "- Characteristic phrases, expressions, or habits\n"
                "- Level of detail and structure (brief and punchy vs. thorough)\n\n"
                f"Writing samples:\n---\n{samples}\n---\n\n"
                "Output ONLY the bullet-point style guide. No preamble or labels."
            ),
        }],
    )

    style_summary = response.content[0].text.strip()
    db.save_user_profile({"style_summary": style_summary})
    return style_summary
