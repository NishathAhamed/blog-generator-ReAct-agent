from __future__ import annotations

import json
import os
import re
from typing import Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_groq import ChatGroq

from state import AgentState
from tools import TOOLS, fetch_image

MAX_TOTAL_TOOL_STEPS = 6
MAX_STEPS_IN_CONTEXT = 3
MAX_IMAGES = 2


def _truncate(s: str, n: int = 1200) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "...[truncated]"


def _scratchpad(steps: list[tuple[AgentAction, str]]) -> str:
    recent = steps[-MAX_STEPS_IN_CONTEXT:] if steps else []
    blocks = []
    for a, obs in recent:
        blocks.append(
            f"Action: {a.tool}\n"
            f"Action Input: {json.dumps(a.tool_input, ensure_ascii=False)}\n"
            f"Observation: {_truncate(obs)}"
        )
    return "\n\n".join(blocks) if blocks else "(none)"


def _is_valid_doc_spec(d: dict) -> bool:
    if not isinstance(d, dict):
        return False
    if not isinstance(d.get("title"), str) or not d["title"].strip():
        return False
    if "subtitle" not in d:
        return False
    if not isinstance(d.get("sections"), list) or len(d["sections"]) == 0:
        return False
    if not isinstance(d.get("references", []), list):
        return False
    for s in d["sections"]:
        if not isinstance(s, dict):
            return False
        if not isinstance(s.get("heading"), str):
            return False
        if not isinstance(s.get("paragraphs"), list) or not s["paragraphs"]:
            return False
        if not isinstance(s.get("images", []), list):
            return False
    return True


def _extract_json_object(text: str) -> Optional[str]:
    m = re.search(r"(\{[\s\S]*\})", text)
    return m.group(1).strip() if m else None


def _force_final_doc(state: AgentState) -> AgentFinish:
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model, temperature=0, max_tokens=1200, timeout=60, max_retries=0)

    assets_brief = [
        {"asset_id": a["asset_id"], "source_url": a["source_url"]} for a in state.get("assets", [])
    ]

    prompt = f"""
Return ONLY valid JSON (no markdown). Must match schema exactly.

Schema:
{{
  "title": "string",
  "subtitle": "string",
  "sections": [
    {{
      "heading": "string",
      "paragraphs": ["string", "string"],
      "images": [{{"asset_id":"string", "caption":"string"}}]
    }}
  ],
  "references": [{{"title":"string","url":"string"}}]
}}

Topic: {state["topic"]}

Available assets:
{json.dumps(assets_brief, ensure_ascii=False)}

Recent tool observations:
{_scratchpad(state.get("intermediate_steps", []))}

Rules:
- Minimum 3 sections.
- If assets exist, reference up to 2 images using EXACT asset_id values above.
- Add 3–6 references from the search results.
"""
    msg = llm.invoke(prompt).content
    raw = _extract_json_object(str(msg)) or str(msg).strip()
    try:
        d = json.loads(raw)
    except Exception:
        d = {}

    if not _is_valid_doc_spec(d):
        d = {
            "title": state["topic"],
            "subtitle": "",
            "sections": [
                {
                    "heading": "Draft",
                    "paragraphs": ["Could not generate a full blog spec reliably. Please retry."],
                    "images": [],
                }
            ],
            "references": [],
        }
        raw = json.dumps(d, ensure_ascii=False)

    return AgentFinish(return_values={"output": raw}, log=str(msg))


def _parse_model_output_to_action_or_finish(text: str) -> AgentAction | AgentFinish:
    t = text.strip()

    if t.startswith("{") and t.endswith("}"):
        d = json.loads(t)
        if _is_valid_doc_spec(d):
            return AgentFinish(return_values={"output": t}, log=text)
        raise ValueError("JSON returned but invalid doc_spec")

    m_final = re.search(r"Final Answer:\s*(\{[\s\S]*\})\s*$", text, re.I)
    if m_final:
        raw = m_final.group(1).strip()
        d = json.loads(raw)
        if not _is_valid_doc_spec(d):
            raise ValueError("Final Answer JSON invalid doc_spec")
        return AgentFinish(return_values={"output": raw}, log=text)

    m_act = re.search(r"Action:\s*(\w+)\s*[\r\n]+Action Input:\s*([\s\S]+)$", text, re.I)
    if not m_act:
        raise ValueError(f"Could not parse model output:\n{text}")

    tool = m_act.group(1).strip()
    tool_input_raw = m_act.group(2).strip()

    try:
        tool_input = json.loads(tool_input_raw)
    except Exception:
        tool_input = tool_input_raw

    return AgentAction(tool=tool, tool_input=tool_input, log=text)


def reason_node(state: AgentState):
    steps = state.get("intermediate_steps", [])
    assets = state.get("assets", [])

    # Bulletproof: bootstrap a search if nothing has happened yet
    if not steps and not assets:
        q = f"{state['topic']} diagram pipeline png"
        return {"agent_outcome": AgentAction(tool="web_search", tool_input=q, log="bootstrap web_search")}

    if len(steps) >= MAX_TOTAL_TOOL_STEPS:
        return {"agent_outcome": _force_final_doc(state)}

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(model=model, temperature=0, max_tokens=600, timeout=60, max_retries=0)

    assets_brief = [{"asset_id": a["asset_id"], "source_url": a["source_url"]} for a in assets]

    prompt = f"""
You are a minimal ReAct agent that produces a Word-blog JSON spec.

Tools: [web_search, fetch_image]

Rules:
- If no assets exist yet, call fetch_image for up to {MAX_IMAGES} image URLs seen in web_search observation.
- Then output Final Answer as JSON matching the schema.
- If assets exist, you MUST use EXACT asset_id values from "Available assets".

Tool call format:
Action: web_search
Action Input: "your query"

Action: fetch_image
Action Input: "https://...png"

Final:
Final Answer: {{...doc_spec json...}}

Topic: {state["topic"]}

Available assets:
{json.dumps(assets_brief, ensure_ascii=False)}

Previous steps:
{_scratchpad(steps)}
"""
    text = str(llm.invoke(prompt).content)
    try:
        outcome = _parse_model_output_to_action_or_finish(text)
        return {"agent_outcome": outcome}
    except Exception:
        return {"agent_outcome": _force_final_doc(state)}


def act_node(state: AgentState):
    outcome = state["agent_outcome"]
    if outcome is None or isinstance(outcome, AgentFinish):
        return {}

    action: AgentAction = outcome
    tool_name = action.tool
    tool_input = action.tool_input

    tool_map = {t.name: t for t in TOOLS}
    tool = tool_map.get(tool_name)

    if tool is None:
        return {"intermediate_steps": [(action, f"Tool '{tool_name}' not found")]}

    # Normalize tool inputs
    if tool_name in {"web_search", "fetch_image"} and not isinstance(tool_input, str):
        if isinstance(tool_input, dict) and "query" in tool_input:
            tool_input = tool_input["query"]
        elif isinstance(tool_input, dict) and "url" in tool_input:
            tool_input = tool_input["url"]
        else:
            tool_input = str(tool_input)

    try:
        raw_obs = tool.invoke(tool_input)
        obs = _truncate(raw_obs)
    except Exception as e:
        return {"intermediate_steps": [(action, f"Tool '{tool_name}' failed: {e}")]}

    steps_update: list[tuple[AgentAction, str]] = [(action, obs)]
    assets_update = []

    # Auto-download top images right after web_search
    if tool_name == "web_search":
        try:
            data = json.loads(raw_obs) if isinstance(raw_obs, str) else raw_obs
            imgs = (data.get("images") or [])[:MAX_IMAGES]
            downloaded = []

            for im in imgs:
                url = (im or {}).get("url")
                if not url:
                    continue
                fa = AgentAction(tool="fetch_image", tool_input=url, log=f"auto fetch {url}")
                asset_json = fetch_image.invoke(url)
                asset = json.loads(asset_json)
                assets_update.append(asset)
                downloaded.append(asset["asset_id"])
                steps_update.append((fa, f"downloaded {asset['asset_id']} from {url}"))

            if downloaded:
                steps_update[0] = (steps_update[0][0], steps_update[0][1] + f"\nDownloaded assets: {downloaded}")

        except Exception as e:
            steps_update[0] = (steps_update[0][0], steps_update[0][1] + f"\n(auto image download skipped: {e})")

    # Register assets if tool was fetch_image
    if tool_name == "fetch_image":
        try:
            asset = json.loads(raw_obs)
            assets_update.append(asset)
        except Exception:
            pass

    return {"intermediate_steps": steps_update, "assets": assets_update}
