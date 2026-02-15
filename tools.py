from __future__ import annotations

import json
import uuid
from io import BytesIO
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from PIL import Image

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"
ASSETS_DIR = OUT_DIR / "assets"
OUT_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)

_tavily = TavilySearch(
    max_results=5,
    search_depth="basic",
    include_images=True,
    include_image_descriptions=True,
)

ALLOWED_IMAGE_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024  # 5MB


@tool("web_search", description="Search the web using Tavily. Input: query string. Output: compact JSON {results, images}.")
def web_search(query: str) -> str:
    """Search Tavily and return compact JSON with results and image URLs."""
    data = _tavily.invoke({"query": query})  # Tavily expects {"query": "..."} :contentReference[oaicite:3]{index=3}

    out = {"query": query, "results": [], "images": []}

    if isinstance(data, dict):
        for r in (data.get("results") or [])[:5]:
            out["results"].append(
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": (r.get("content") or "")[:600],
                }
            )
        for im in (data.get("images") or [])[:6]:
            out["images"].append({"url": im.get("url"), "description": im.get("description") or ""})

    return json.dumps(out, ensure_ascii=False)


@tool("fetch_image", description="Download image URL (PNG/JPEG/WebP) to output/assets. Returns JSON {asset_id,path,source_url}.")
def fetch_image(url: str) -> str:
    """Download image URL to local disk for python-docx embedding."""
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if ctype not in ALLOWED_IMAGE_MIME:
        raise ValueError(f"Unsupported image type: {ctype}")

    content = r.content
    if len(content) > MAX_BYTES:
        raise ValueError("Image too large (>5MB)")

    asset_id = f"img_{uuid.uuid4().hex[:8]}"

    if ctype == "image/webp":
        img = Image.open(BytesIO(content)).convert("RGB")
        path = ASSETS_DIR / f"{asset_id}.png"
        img.save(path, format="PNG")
    else:
        ext = ".jpg" if ctype == "image/jpeg" else ".png"
        path = ASSETS_DIR / f"{asset_id}{ext}"
        path.write_bytes(content)

    # RETURN ABSOLUTE PATH (prevents CWD mismatch)
    abs_path = str(path.resolve())
    return json.dumps({"asset_id": asset_id, "path": abs_path, "source_url": url}, ensure_ascii=False)


TOOLS = [web_search, fetch_image]
