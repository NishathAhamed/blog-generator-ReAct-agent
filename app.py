# app.py
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile

from dotenv import load_dotenv
from langchain_core.agents import AgentFinish
from langchain_groq import ChatGroq

load_dotenv()

from graph import app
from renderer import render_docx


def _list_embedded_media(docx_path: str) -> list[str]:
    with zipfile.ZipFile(docx_path) as z:
        return [n for n in z.namelist() if n.startswith("word/media/")]

def _count_words(doc_spec: dict) -> int:
    def wc(s: str) -> int:
        return len(str(s).split())
    total = 0
    total += wc(doc_spec.get("title", ""))
    total += wc(doc_spec.get("subtitle", ""))
    for sec in doc_spec.get("sections", []) or []:
        total += wc(sec.get("heading", ""))
        for p in sec.get("paragraphs", []) or []:
            total += wc(p)
        for img in sec.get("images", []) or []:
            total += wc(img.get("caption", ""))
    return total

def _extract_json_object(text: str) -> str:
    m = re.search(r"(\{[\s\S]*\})", text)
    return (m.group(1) if m else text).strip()

def _is_valid_doc_spec(d: dict) -> bool:
    return (
        isinstance(d, dict)
        and isinstance(d.get("title"), str) and d["title"].strip()
        and "subtitle" in d
        and isinstance(d.get("sections"), list) and len(d["sections"]) > 0
        and isinstance(d.get("references", []), list)
    )

def _expand_to_target(doc_spec: dict, target_words: int, assets_brief: list[dict]) -> dict:
    # LLM word counts aren’t exact; use range. :contentReference[oaicite:1]{index=1}
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model, temperature=0, max_tokens=2000, timeout=60, max_retries=0)

    prompt = f"""
You must return ONLY valid JSON (no markdown). Expand the blog to be ~{target_words} words (±10%).
Keep the same schema.

Schema:
{{
  "title": "string",
  "subtitle": "string",
  "sections": [{{"heading":"string","paragraphs":["..."],"images":[{{"asset_id":"string","caption":"string"}}]}}],
  "references": [{{"title":"string","url":"string"}}]
}}

Rules:
- Make content longer by adding paragraphs and detail (not fluff).
- Aim for 4–6 sections.
- Each section should have 2–5 paragraphs, ~60–110 words each.
- If assets exist, reference up to 2 images using EXACT asset_id values:
{json.dumps(assets_brief, ensure_ascii=False)}

Here is the current JSON to expand:
{json.dumps(doc_spec, ensure_ascii=False)}
"""
    raw = _extract_json_object(str(llm.invoke(prompt).content))
    d = json.loads(raw)
    if not _is_valid_doc_spec(d):
        return doc_spec
    return d


def run(topic: str, target_words: int) -> str:
    result = app.invoke(
        {
            "topic": topic,
            "target_words": target_words,     # NEW
            "agent_outcome": None,
            "intermediate_steps": [],
            "assets": [],
        },
        config={"recursion_limit": 30},
    )

    final = result.get("agent_outcome")
    if not isinstance(final, AgentFinish):
        raise RuntimeError(f"Agent did not finish. outcome={type(final)}")

    raw = final.return_values.get("output", "{}")
    doc_spec = json.loads(raw)

    assets = result.get("assets", [])
    assets_by_id = {a["asset_id"]: a for a in assets}
    assets_brief = [{"asset_id": a["asset_id"], "source_url": a["source_url"]} for a in assets]

    # Bulletproof length enforcement (max 2 expansions)
    for _ in range(2):
        wc = _count_words(doc_spec)
        if wc >= int(target_words * 0.9):
            break
        doc_spec = _expand_to_target(doc_spec, target_words, assets_brief)

    out_path = os.path.join("output", "blog.docx")
    render_docx(doc_spec, assets_by_id, out_path)

    print("Saved:", out_path)
    print("Approx words:", _count_words(doc_spec))
    print("Embedded images:", _list_embedded_media(out_path))
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topic", default="Retrieval-Augmented Generation (RAG)")
    p.add_argument("--words", type=int, default=1500)   # user-controlled
    args = p.parse_args()

    run(args.topic, args.words)
