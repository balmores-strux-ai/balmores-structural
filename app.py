import os
import re
import json
import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI

from fem_core import (
    parse_nodes_text,
    parse_members_text,
    parse_supports_text,
    parse_nodal_loads_text,
    parse_section_setup_text,
    update_nodes,
    analyze_structure,
    get_ai_context_from_result,
    charts_payload_from_result,
    etabs_style_export_text,
    report_sections,
    format_immediate_chat_results,
)
from brain_model import brain_recommendation_text, brain_status_message, brain_config_public

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

app = FastAPI(title="BALMORES STRUX AI")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

PROJECT_STATE: Dict[str, Any] = {
    "nodes": {},
    "members": {},
    "supports": {},
    "nodal_loads": {},
    "family_sections": {
        "beam": "W360x44",
        "column": "W310x60",
        "brace": "HSS203x203x9.5",
    },
    "building_code": "US",
    "materials": {
        "fc_MPa": None,
        "fy_MPa": None,
        "sbc_kPa": None,
    },
    "last_result": None,
    "last_report": None,
    "last_charts": None,
    "last_etabs_export": "",
    "messages": [],
    "history": [],
}


def push_history():
    PROJECT_STATE["history"].append(copy.deepcopy({
        "nodes": PROJECT_STATE["nodes"],
        "members": PROJECT_STATE["members"],
        "supports": PROJECT_STATE["supports"],
        "nodal_loads": PROJECT_STATE["nodal_loads"],
        "family_sections": PROJECT_STATE["family_sections"],
        "building_code": PROJECT_STATE["building_code"],
        "materials": dict(PROJECT_STATE["materials"]),
        "last_result": PROJECT_STATE["last_result"],
        "last_report": PROJECT_STATE["last_report"],
        "last_charts": PROJECT_STATE["last_charts"],
        "last_etabs_export": PROJECT_STATE["last_etabs_export"],
    }))
    if len(PROJECT_STATE["history"]) > 50:
        PROJECT_STATE["history"].pop(0)


QUOTA_ERROR_MSG = (
    "OpenAI API quota exceeded. Add billing at platform.openai.com or wait for monthly reset. "
    "Try the Sample button to load a 4-storey demo and run FEM without the API."
)


def _fallback_build_from_text(text: str) -> Optional[Dict[str, str]]:
    """Parse 'N storey Xm x Ym' or 'N storey bay x X m bay y Y m' style without OpenAI."""
    text = text.lower().strip()
    n_story = None
    bay_x = 6.0
    bay_y = 12.0

    m = re.search(
        r"(\d+)\s*storey\s*(?:building|steel)?\s*"
        r"(?:(\d+(?:\.\d+)?)\s*[mx×]\s*(\d+(?:\.\d+)?)|"
        r"(\d+(?:\.\d+)?)\s*m?\s*[mx×]\s*(\d+(?:\.\d+)?)\s*m?)",
        text,
        re.I,
    )
    if m:
        g = m.groups()
        n_story = int(g[0])
        bay_x = float(g[1] or g[3] or 6)
        bay_y = float(g[2] or g[4] or 12)

    if n_story is None:
        m2 = re.search(
            r"(\d+)\s*storey\s*(?:steel\s+)?building\s+(?:bay\s+)?"
            r"x\s*(\d+(?:\.\d+)?)\s*m\s+(?:bay\s+)?y\s*(\d+(?:\.\d+)?)\s*m",
            text,
            re.I,
        )
        if m2:
            n_story = int(m2.group(1))
            bay_x = float(m2.group(2))
            bay_y = float(m2.group(3))
        else:
            m3 = re.search(
                r"(\d+)\s*storey.*?bay\s*x\s*(\d+(?:\.\d+)?)\s*m.*?bay\s*y\s*(\d+(?:\.\d+)?)\s*m",
                text,
                re.I | re.DOTALL,
            )
            if m3:
                n_story = int(m3.group(1))
                bay_x = float(m3.group(2))
                bay_y = float(m3.group(3))

    if n_story is None:
        return None
    story_h = 4.0
    n_x, n_y = 2, 2
    if bay_x > bay_y:
        n_x, n_y = 2, 3
    elif bay_y > bay_x:
        n_x, n_y = 3, 2
    xs = [i * bay_x for i in range(n_x + 1)]
    ys = [i * bay_y for i in range(n_y + 1)]
    n_pts = (n_x + 1) * (n_y + 1)
    nid = 1
    node_lines = []
    for z in [i * story_h for i in range(n_story + 1)]:
        for y in ys:
            for x in xs:
                node_lines.append(f"{nid}({x} {y} {z})")
                nid += 1
    nodes_str = " ".join(node_lines)
    col_per_plan = n_pts
    mem_lines = []
    mid = 1
    for lev in range(n_story):
        base = lev * col_per_plan + 1
        next_base = (lev + 1) * col_per_plan + 1
        for i in range(col_per_plan):
            mem_lines.append(f"{mid}({base + i} {next_base + i})")
            mid += 1
    nx1, ny1 = n_x + 1, n_y + 1
    for lev in range(n_story + 1):
        base = lev * col_per_plan + 1
        for row in range(ny1):
            for col in range(n_x):
                a = base + row * (nx1) + col
                b = a + 1
                mem_lines.append(f"{mid}({a} {b})")
                mid += 1
        for col in range(nx1):
            for row in range(n_y):
                a = base + row * (nx1) + col
                b = a + (nx1)
                mem_lines.append(f"{mid}({a} {b})")
                mid += 1
    members_str = " ".join(mem_lines)
    supports_str = " ".join(f"{i} fixed" for i in range(1, col_per_plan + 1))
    load_nodes = [n_story * col_per_plan + i for i in range(1, col_per_plan + 1)]
    loads_str = " ".join(f"{n}(0 0 -40 0 0 0)" for n in load_nodes)
    sections_str = "beam W360x44 column W310x60 brace HSS203x203x9.5"
    notes = f"Built-in parser: {n_story} storey, {bay_x}m x {bay_y}m, {col_per_plan} cols/level. No OpenAI used."
    return {
        "nodes": nodes_str,
        "members": members_str,
        "supports": supports_str,
        "loads": loads_str,
        "sections": sections_str,
        "notes": notes,
    }


def _append_message(role: str, content: str):
    PROJECT_STATE["messages"].append({"role": role, "content": content})
    if len(PROJECT_STATE["messages"]) > 40:
        PROJECT_STATE["messages"] = PROJECT_STATE["messages"][-40:]


def _refresh_derived_outputs():
    """Recompute report, charts, ETABS text from last_result + materials."""
    res = PROJECT_STATE["last_result"]
    mats = PROJECT_STATE["materials"]
    brain_line = brain_recommendation_text(res) if res and res.get("ok") else ""
    PROJECT_STATE["last_report"] = report_sections(res, mats, brain_line)
    PROJECT_STATE["last_charts"] = charts_payload_from_result(res)
    if res and res.get("ok"):
        inp = res.get("inputs") or {}
        PROJECT_STATE["last_etabs_export"] = etabs_style_export_text(
            inp.get("nodes") or {},
            inp.get("members") or {},
            inp.get("supports") or {},
            inp.get("nodal_loads") or {},
            inp.get("family_sections") or PROJECT_STATE["family_sections"],
        )
    else:
        PROJECT_STATE["last_etabs_export"] = ""


def read_response_text(response) -> str:
    text = getattr(response, "output_text", "")
    if text:
        return text.strip()

    chunks = []
    for item in getattr(response, "output", []):
        if getattr(item, "type", "") == "message":
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    chunks.append(content.text)
    return "".join(chunks).strip()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _export_engineering_bundle() -> Dict[str, Any]:
    def nodes_map(d):
        return {str(k): [float(x) for x in v] for k, v in d.items()}

    def mem_map(d):
        return {str(k): [int(x) for x in v] for k, v in d.items()}

    loads = {}
    for k, v in PROJECT_STATE["nodal_loads"].items():
        loads[str(k)] = [float(x) for x in v]

    return {
        "format": "balmores-strux-pack",
        "version": 1,
        "building_code": PROJECT_STATE.get("building_code"),
        "materials": dict(PROJECT_STATE.get("materials") or {}),
        "nodes": nodes_map(PROJECT_STATE.get("nodes") or {}),
        "members": mem_map(PROJECT_STATE.get("members") or {}),
        "supports": {str(k): v for k, v in (PROJECT_STATE.get("supports") or {}).items()},
        "nodal_loads": loads,
        "family_sections": dict(PROJECT_STATE.get("family_sections") or {}),
        "fem_result": PROJECT_STATE.get("last_result"),
        "report": PROJECT_STATE.get("last_report"),
        "charts": PROJECT_STATE.get("last_charts"),
        "etabs_text": PROJECT_STATE.get("last_etabs_export") or "",
        "brain": brain_config_public(),
    }


@app.get("/api/state")
async def get_state():
    return {
        "ok": True,
        "project": PROJECT_STATE,
        "brain_status": brain_status_message(),
        "brain": brain_config_public(),
    }


@app.get("/api/export-pack")
async def export_pack():
    """One JSON file: geometry, loads, FEM output, ETABS text — for records or your own scripts."""
    return JSONResponse(_export_engineering_bundle())


@app.post("/api/settings")
async def update_settings(payload: Dict[str, Any]):
    code = (payload.get("building_code") or "").strip().upper()
    if code:
        PROJECT_STATE["building_code"] = code
    mats = payload.get("materials")
    if isinstance(mats, dict):
        for k in ("fc_MPa", "fy_MPa", "sbc_kPa"):
            if k in mats:
                v = mats[k]
                PROJECT_STATE["materials"][k] = None if v in ("", None) else float(v)
    if PROJECT_STATE.get("last_result"):
        _refresh_derived_outputs()
    return {"ok": True, "project": PROJECT_STATE}


@app.post("/api/undo")
async def undo_state():
    if not PROJECT_STATE["history"]:
        return {"ok": False, "message": "Nothing to undo."}

    prev = PROJECT_STATE["history"].pop()
    PROJECT_STATE["nodes"] = prev["nodes"]
    PROJECT_STATE["members"] = prev["members"]
    PROJECT_STATE["supports"] = prev["supports"]
    PROJECT_STATE["nodal_loads"] = prev["nodal_loads"]
    PROJECT_STATE["family_sections"] = prev["family_sections"]
    PROJECT_STATE["building_code"] = prev.get("building_code", PROJECT_STATE["building_code"])
    PROJECT_STATE["materials"] = dict(prev.get("materials", PROJECT_STATE["materials"]))
    PROJECT_STATE["last_result"] = prev["last_result"]
    PROJECT_STATE["last_report"] = prev.get("last_report")
    PROJECT_STATE["last_charts"] = prev.get("last_charts")
    PROJECT_STATE["last_etabs_export"] = prev.get("last_etabs_export", "")

    return {"ok": True, "message": "Undo completed.", "project": PROJECT_STATE}


@app.post("/api/model")
async def model_command(payload: Dict[str, Any]):
    cmd = payload.get("command", "")
    text = payload.get("text", "")

    try:
        if cmd == "set_nodes":
            push_history()
            PROJECT_STATE["nodes"] = parse_nodes_text(text)
            return {"ok": True, "message": "Nodes accepted.", "project": PROJECT_STATE}

        if cmd == "set_members":
            push_history()
            PROJECT_STATE["members"] = parse_members_text(text, PROJECT_STATE["nodes"])
            return {"ok": True, "message": "Members accepted.", "project": PROJECT_STATE}

        if cmd == "set_supports":
            push_history()
            PROJECT_STATE["supports"] = parse_supports_text(text, PROJECT_STATE["nodes"])
            return {"ok": True, "message": "Supports accepted.", "project": PROJECT_STATE}

        if cmd == "set_loads":
            push_history()
            PROJECT_STATE["nodal_loads"] = parse_nodal_loads_text(text, PROJECT_STATE["nodes"])
            return {"ok": True, "message": "Nodal loads accepted.", "project": PROJECT_STATE}

        if cmd == "set_sections":
            push_history()
            PROJECT_STATE["family_sections"] = parse_section_setup_text(text)
            return {"ok": True, "message": "Sections accepted.", "project": PROJECT_STATE}

        if cmd == "edit_nodes":
            push_history()
            PROJECT_STATE["nodes"] = update_nodes(PROJECT_STATE["nodes"], text)
            return {"ok": True, "message": "Nodes updated.", "project": PROJECT_STATE}

        if cmd == "run_sample":
            push_history()
            PROJECT_STATE["nodes"] = parse_nodes_text(
                "1(0 0 0) 2(6 0 0) 3(0 0 4) 4(6 0 4) "
                "5(0 8 0) 6(6 8 0) 7(0 8 4) 8(6 8 4)"
            )
            PROJECT_STATE["members"] = parse_members_text(
                "1(1 3) 2(2 4) 3(5 7) 4(6 8) 5(1 2) 6(3 4) 7(5 6) 8(7 8) 9(1 5) 10(2 6) 11(3 7) 12(4 8)",
                PROJECT_STATE["nodes"]
            )
            PROJECT_STATE["supports"] = parse_supports_text(
                "1 fixed 2 fixed 5 fixed 6 fixed",
                PROJECT_STATE["nodes"]
            )
            PROJECT_STATE["nodal_loads"] = parse_nodal_loads_text(
                "3(0 0 -50 0 0 0) 4(0 0 -50 0 0 0) 7(0 0 -50 0 0 0) 8(0 0 -50 0 0 0)",
                PROJECT_STATE["nodes"]
            )
            PROJECT_STATE["family_sections"] = {
                "beam": "W360x44",
                "column": "W310x60",
                "brace": "HSS203x203x9.5",
            }
            return {"ok": True, "message": "Sample model loaded.", "project": PROJECT_STATE}

        if cmd == "run_fem":
            push_history()
            result = analyze_structure(
                PROJECT_STATE["nodes"],
                PROJECT_STATE["members"],
                PROJECT_STATE["supports"],
                PROJECT_STATE["nodal_loads"],
                PROJECT_STATE["family_sections"],
                building_code=PROJECT_STATE.get("building_code") or "US",
            )
            PROJECT_STATE["last_result"] = result
            _refresh_derived_outputs()
            return {
                "ok": result.get("ok", False),
                "message": result.get("message", "Run finished."),
                "project": PROJECT_STATE
            }

        return {"ok": False, "message": "Unknown command."}

    except Exception as e:
        return {"ok": False, "message": str(e)}


@app.post("/api/chat")
async def chat(payload: Dict[str, Any]):
    if client is None:
        return {"ok": False, "message": "OPENAI_API_KEY is missing on the server."}

    user_message = payload.get("message", "").strip()
    if not user_message:
        return {"ok": False, "message": "Empty message."}

    model_context = get_ai_context_from_result(PROJECT_STATE["last_result"])
    mats = PROJECT_STATE.get("materials") or {}
    mat_ctx = (
        f"User materials (sidebar): fc={mats.get('fc_MPa')} MPa, fy={mats.get('fy_MPa')} MPa, "
        f"SBC={mats.get('sbc_kPa')} kPa. Building code selection: {PROJECT_STATE.get('building_code', 'US')}."
    )
    rep = PROJECT_STATE.get("last_report") or {}
    if rep:
        model_context += (
            "\nLatest app report summary:\n"
            f"{rep.get('summary', '')}\n{rep.get('conclusion', '')}"
        )

    system_prompt = (
        "You are BALMORES STRUX AI, a structural engineering assistant inside a web app. "
        "Answer clearly, practically, and professionally. "
        "If the user asks about the current model, use the provided model context. "
        "Do not invent solved checks that the app did not compute. "
        "Encourage verifying results in ETABS for final code compliance."
    )

    incoming = payload.get("messages")
    if isinstance(incoming, list) and incoming:
        thread = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))} for m in incoming][-20:]
    else:
        thread = list(PROJECT_STATE.get("messages") or [])[-20:]

    input_items = [
        {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
        {"role": "system", "content": [{"type": "input_text", "text": model_context}]},
        {"role": "system", "content": [{"type": "input_text", "text": mat_ctx}]},
    ]
    for m in thread:
        role = m.get("role", "user")
        if role not in ("user", "assistant"):
            role = "user"
        input_items.append({
            "role": role,
            "content": [{"type": "input_text", "text": m.get("content", "")}],
        })
    input_items.append({
        "role": "user",
        "content": [{"type": "input_text", "text": user_message}],
    })

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=input_items,
        )
        answer = read_response_text(response) or "No response text returned."
        _append_message("user", user_message)
        _append_message("assistant", answer)
        return {"ok": True, "message": answer, "project": PROJECT_STATE}

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "insufficient_quota" in err.lower():
            return {"ok": False, "message": QUOTA_ERROR_MSG}
        return {"ok": False, "message": err}


def _apply_settings_from_payload(payload: Dict[str, Any]):
    code = (payload.get("building_code") or "").strip().upper()
    if code:
        PROJECT_STATE["building_code"] = code
    mats = payload.get("materials")
    if isinstance(mats, dict):
        for k in ("fc_MPa", "fy_MPa", "sbc_kPa"):
            if k in mats and mats[k] not in ("", None):
                try:
                    PROJECT_STATE["materials"][k] = float(mats[k])
                except (TypeError, ValueError):
                    pass


@app.post("/api/nlm")
async def natural_language_model(payload: Dict[str, Any]):
    if client is None:
        return {"ok": False, "message": "OPENAI_API_KEY is missing on the server."}

    user_message = payload.get("message", "").strip()
    if not user_message:
        return {"ok": False, "message": "Empty message."}

    _apply_settings_from_payload(payload)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nodes": {"type": "string"},
            "members": {"type": "string"},
            "supports": {"type": "string"},
            "loads": {"type": "string"},
            "sections": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["nodes", "members", "supports", "loads", "sections", "notes"]
    }

    mats = PROJECT_STATE["materials"]
    prompt = (
        "Convert the user's structural description into a buildable 3D steel frame model.\n"
        "Use only these output fields:\n"
        "nodes, members, supports, loads, sections, notes.\n\n"
        "Formatting rules:\n"
        "nodes example: 1(0 0 0) 2(6 0 0) 3(0 0 4)\n"
        "members example: 1(1 3) 2(2 4) 3(1 2)\n"
        "supports example: 1 fixed 2 fixed\n"
        "loads example: 3(0 0 -50 0 0 0) 4(0 0 -50 0 0 0)\n"
        "sections example: beam W360x44 column W310x60 brace HSS203x203x9.5\n"
        "Coordinates are global X Y Z in meters (similar spirit to STAAD/ETABS joint coordinates).\n"
        "Interpret bay spacing in X and Y, building footprint, and number of stories/height from the user text.\n"
        "Use nodal loads in kN / kNm as 6 DOF vectors Fx Fy Fz Mx My Mz per loaded joint.\n"
        "Library sections ONLY: W360x44 W310x60 W410x60 W460x74 W530x85 HSS203x203x9.5\n"
        "If the user mentions concrete fc, steel fy, or soil SBC, summarize how you interpreted them in notes "
        "(the web app tracks them separately for reporting; still keep loads reasonable).\n"
        f"Selected building code tag for drift context: {PROJECT_STATE.get('building_code', 'US')}.\n"
        f"Sidebar materials: fc={mats.get('fc_MPa')} MPa, fy={mats.get('fy_MPa')} MPa, SBC={mats.get('sbc_kPa')} kPa.\n"
        "If information is incomplete, make reasonable structural assumptions and explain them briefly in notes."
    )

    user_block = user_message

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_block}]
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "nlm_model_output",
                    "strict": True,
                    "schema": schema
                }
            }
        )

        raw = read_response_text(response)
        parsed = json.loads(raw)

        push_history()

        PROJECT_STATE["nodes"] = parse_nodes_text(parsed["nodes"])
        PROJECT_STATE["members"] = parse_members_text(parsed["members"], PROJECT_STATE["nodes"])
        PROJECT_STATE["supports"] = parse_supports_text(parsed["supports"], PROJECT_STATE["nodes"])
        PROJECT_STATE["nodal_loads"] = parse_nodal_loads_text(parsed["loads"], PROJECT_STATE["nodes"])
        PROJECT_STATE["family_sections"] = parse_section_setup_text(parsed["sections"])

        _append_message("user", user_message)
        _append_message("assistant", parsed.get("notes") or "Model generated.")

        return {
            "ok": True,
            "message": parsed["notes"],
            "project": PROJECT_STATE,
            "generated": parsed
        }

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "insufficient_quota" in err.lower():
            return {"ok": False, "message": QUOTA_ERROR_MSG}
        return {"ok": False, "message": err}


def _run_fem_and_append():
    """Helper: run FEM, refresh outputs, append ChatGPT-style results to chat."""
    result = analyze_structure(
        PROJECT_STATE["nodes"],
        PROJECT_STATE["members"],
        PROJECT_STATE["supports"],
        PROJECT_STATE["nodal_loads"],
        PROJECT_STATE["family_sections"],
        building_code=PROJECT_STATE.get("building_code") or "US",
    )
    PROJECT_STATE["last_result"] = result
    _refresh_derived_outputs()
    if result.get("ok"):
        block = format_immediate_chat_results(result, PROJECT_STATE.get("materials"))
        _append_message("assistant", block)
    else:
        _append_message("assistant", f"FEM: {result.get('message', '')[:500]}")
    return result


@app.post("/api/build-analyze")
async def build_and_analyze(payload: Dict[str, Any]):
    """NLM → FEM → charts/report/export. Uses built-in parser for 'N storey Xm x Ym' when API fails."""
    user_message = payload.get("message", "").strip()
    if not user_message:
        return {"ok": False, "message": "Empty message."}

    _apply_settings_from_payload(payload)
    auto_run = payload.get("auto_analyze", True)
    parsed = None

    fallback = _fallback_build_from_text(user_message)
    if fallback:
        try:
            push_history()
            PROJECT_STATE["nodes"] = parse_nodes_text(fallback["nodes"])
            PROJECT_STATE["members"] = parse_members_text(fallback["members"], PROJECT_STATE["nodes"])
            PROJECT_STATE["supports"] = parse_supports_text(fallback["supports"], PROJECT_STATE["nodes"])
            PROJECT_STATE["nodal_loads"] = parse_nodal_loads_text(fallback["loads"], PROJECT_STATE["nodes"])
            PROJECT_STATE["family_sections"] = parse_section_setup_text(fallback["sections"])
            _append_message("user", user_message)
            _append_message("assistant", fallback["notes"])
            parsed = fallback
            if auto_run:
                _run_fem_and_append()
            return {
                "ok": True,
                "message": fallback["notes"],
                "fem_summary": PROJECT_STATE.get("last_result", {}).get("message", ""),
                "project": PROJECT_STATE,
                "generated": parsed,
                "brain_status": brain_status_message(),
            }
        except Exception as e:
            return {"ok": False, "message": f"Built-in parser error: {e}"}

    if client is None:
        return {"ok": False, "message": "OPENAI_API_KEY missing. Try phrases like '4 storey 6m x 12m' — uses built-in parser."}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "nodes": {"type": "string"},
            "members": {"type": "string"},
            "supports": {"type": "string"},
            "loads": {"type": "string"},
            "sections": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["nodes", "members", "supports", "loads", "sections", "notes"]
    }

    mats = PROJECT_STATE["materials"]
    prompt = (
        "Convert the user's structural description into a buildable 3D steel frame model.\n"
        "Output JSON only per schema with nodes, members, supports, loads, sections, notes.\n"
        "Coordinates: global X Y Z in meters (STAAD/ETABS-style joints).\n"
        "Respect bay X, bay Y, building height, and grids described by the user.\n"
        "sections must use ONLY: W360x44 W310x60 W410x60 W460x74 W530x85 HSS203x203x9.5\n"
        f"Building code context: {PROJECT_STATE.get('building_code', 'US')}. "
        f"Materials context: fc={mats.get('fc_MPa')} MPa, fy={mats.get('fy_MPa')} MPa, SBC={mats.get('sbc_kPa')} kPa.\n"
        "Use nodal loads only (kN, kNm). Explain assumptions in notes."
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_message}]},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "build_analyze_model",
                    "strict": True,
                    "schema": schema
                }
            }
        )

        raw = read_response_text(response)
        parsed = json.loads(raw)

        push_history()

        PROJECT_STATE["nodes"] = parse_nodes_text(parsed["nodes"])
        PROJECT_STATE["members"] = parse_members_text(parsed["members"], PROJECT_STATE["nodes"])
        PROJECT_STATE["supports"] = parse_supports_text(parsed["supports"], PROJECT_STATE["nodes"])
        PROJECT_STATE["nodal_loads"] = parse_nodal_loads_text(parsed["loads"], PROJECT_STATE["nodes"])
        PROJECT_STATE["family_sections"] = parse_section_setup_text(parsed["sections"])

        _append_message("user", user_message)
        _append_message("assistant", parsed.get("notes") or "Model built.")

        fem_msg = ""
        if auto_run:
            res = _run_fem_and_append()
            fem_msg = res.get("message", "")

        return {
            "ok": True,
            "message": parsed.get("notes", ""),
            "fem_summary": fem_msg,
            "project": PROJECT_STATE,
            "generated": parsed,
            "brain_status": brain_status_message(),
        }

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "insufficient_quota" in err.lower():
            return {"ok": False, "message": QUOTA_ERROR_MSG}
        return {"ok": False, "message": err}