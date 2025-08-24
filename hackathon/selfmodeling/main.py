# mcp_self_host_hardened.py
# A hardened, self-healing, self-evolving single-file MCP server.
# - Locks dynamic markers to prevent corruption.
# - Can repair its own file (normalize markers, migrate orphan blocks, move region).
# - Can install/execute new @mcp.tool() blocks at runtime and persist them.
# - Loads persisted blocks on startup so tools survive restarts.
#
# SECURITY: Any injected code runs with full Python privileges. Use only in trusted environments.

from __future__ import annotations

import sys
import re
import uuid
import logging
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from mcp.server.fastmcp import FastMCP

# ---------------- Logging (stderr only; NEVER stdout in stdio mode) ----------------
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mcp-self-host-hardened")

# ---------------- MCP instance ----------------
mcp = FastMCP(
    "mcp-self-host-hardened",
    instructions=("Hardened self-modifying MCP server. "
                  "Use 'add_tool_block' to add tools, 'repair_self_file' to self-heal, "
                  "'reload_all_blocks' to activate persisted blocks.")
)

# ---------------- Tool registry (names/docs/signatures) ----------------
_REGISTERED_TOOL_NAMES: List[str] = []
_TOOL_META: Dict[str, Dict[str, Any]] = {}   # name -> {doc, signature, origin, blocks: [ids]}
_LOADED_BLOCK_IDS: set[str] = set()          # which blocks have been exec()'d (including static import)

# Canonical dynamic region literals
_LIT_START = "# === DYNAMIC TOOLS START ==="
_LIT_END   = "# === DYNAMIC TOOLS END ==="

# Helper: best-effort notify clients that tools list changed (multi-version compatibility)
def _notify_tools_changed_safe() -> bool:
    notified = False
    # Try methods on FastMCP instance
    for attr in ("notify_tools_changed", "announce_tools_changed", "tools_changed", "announce_tools"):
        fn = getattr(mcp, attr, None)
        if callable(fn):
            try:
                fn()
                notified = True
                log.info("tools-changed notification sent via mcp.%s()", attr)
                break
            except Exception as e:
                log.warning("tools-changed via mcp.%s() failed: %r", attr, e)

    # Try low-level server object (implementation dependent)
    if not notified:
        srv = getattr(mcp, "_server", None) or getattr(mcp, "server", None)
        if srv is not None:
            for attr in ("notify_tools_changed", "announce_tools_changed", "tools_changed"):
                fn = getattr(srv, attr, None)
                if callable(fn):
                    try:
                        fn()
                        notified = True
                        log.info("tools-changed notification sent via server.%s()", attr)
                        break
                    except Exception as e:
                        log.warning("tools-changed via server.%s() failed: %r", attr, e)

    if not notified:
        log.info("No tools-changed notifier available; client may need to re-list tools or reconnect.")
    return notified

# ---------------- Patch @mcp.tool decorator to record metadata ----------------
_orig_tool_decorator_factory = mcp.tool
def _recording_tool_decorator_factory(*dargs, **dkwargs):
    inner = _orig_tool_decorator_factory(*dargs, **dkwargs)
    def _decorator(func):
        name = getattr(func, "__name__", "<anonymous>")
        try:
            sig = str(inspect.signature(func))
        except Exception:
            sig = "(...)"
        doc = (func.__doc__ or "").strip()

        # Record tool meta
        _REGISTERED_TOOL_NAMES.append(name)
        meta = _TOOL_META.get(name, {})
        meta.update({
            "doc": doc,
            "signature": sig,
            "origin": meta.get("origin", "dynamic"),
            "blocks": meta.get("blocks", []),
        })

        # If the function was defined while a block was active, link it
        try:
            bid = None
            g = getattr(func, "__globals__", {}) or {}
            bid = g.get("__MCP_ACTIVE_BLOCK_ID", None)
            if isinstance(bid, str) and bid:
                blist = meta.get("blocks", [])
                if bid not in blist:
                    blist.append(bid)
                meta["blocks"] = blist
                _LOADED_BLOCK_IDS.add(bid)
        except Exception as e:
            log.debug("block-linking failed for %s: %r", name, e)

        _TOOL_META[name] = meta

        # Register into FastMCP
        return inner(func)
    return _decorator

# Patch the decorator so dynamic code is recorded
mcp.tool = _recording_tool_decorator_factory  # type: ignore[attr-defined]

# ---------------- Canonical markers + runtime guard ----------------
SELF_PATH = Path(__file__).resolve()

# DO NOT change these lines; they re-lock the markers at runtime even if a generator corrupts code lines
START_MARK = _LIT_START
END_MARK   = _LIT_END

ENTRY_MARK = "# ---------------- Entry ----------------"
MAIN_GUARD = "if __name__"

BLOCK_START = "# --- BLOCK"
BLOCK_END   = "# --- END ---"

# Block regex (greedy body, non-greedy overall)
_BLOCK_RE = re.compile(
    r"(?P<all>#\s*---\s*BLOCK\s+(?P<id>[A-Za-z0-9_\-]+)\s+START\s*---\s*\n)"
    r"(?P<body>.*?)(#\s*---\s*END\s*---\s*\n)",
    re.DOTALL
)

# ---------------- File IO helpers ----------------
def _read_self_text() -> str:
    return SELF_PATH.read_text(encoding="utf-8")

def _write_self_text(new_text: str) -> None:
    bak = SELF_PATH.with_suffix(SELF_PATH.suffix + ".bak")
    try:
        bak.write_text(_read_self_text(), encoding="utf-8")
    except Exception:
        pass
    SELF_PATH.write_text(new_text, encoding="utf-8")

# ---------------- Dynamic region helpers ----------------
def _normalize_marker_lines(text: str) -> Tuple[str, bool]:
    """Normalize any lines containing marker comments to be exactly canonical."""
    changed = False
    pat_start = re.compile(r'^.*#\s*===\s*DYNAMIC TOOLS START\s*===.*$', re.MULTILINE)
    pat_end   = re.compile(r'^.*#\s*===\s*DYNAMIC TOOLS END\s*===.*$',   re.MULTILINE)
    new_text, c1 = pat_start.subn(_LIT_START, text)
    new_text, c2 = pat_end.subn(_LIT_END, new_text)
    if c1 or c2:
        changed = True
    return new_text, changed

def _ensure_dynamic_region_exists(text: str) -> Tuple[str, bool]:
    """Ensure a clean START/END region exists before ENTRY or MAIN_GUARD."""
    text, normed = _normalize_marker_lines(text)
    insert_idx = text.find(ENTRY_MARK)
    if insert_idx == -1:
        insert_idx = text.find(MAIN_GUARD)
    if insert_idx == -1:
        insert_idx = len(text)

    has_start = (_LIT_START in text)
    has_end   = (_LIT_END in text)
    if has_start and has_end:
        # ensure it's before entry; else move.
        start = text.find(_LIT_START)
        end   = text.find(_LIT_END)
        if start == -1 or end == -1 or end < start:
            block = f"{_LIT_START}\n# (dynamic @mcp.tool() blocks will be inserted here)\n{_LIT_END}\n\n"
            return text[:insert_idx] + block + text[insert_idx:], True
        if start < insert_idx:
            return text, normed
        # Move region before ENTRY
        end_line_end = text.find("\n", end)
        if end_line_end == -1:
            end_line_end = len(text)
        region_text = text[start:end_line_end+1]
        text_wo = text[:start] + text[end_line_end+1:]
        new_text = text_wo[:insert_idx] + region_text + text_wo[insert_idx:]
        return new_text, True

    # No region: create
    block = f"{_LIT_START}\n# (dynamic @mcp.tool() blocks will be inserted here)\n{_LIT_END}\n\n"
    return text[:insert_idx] + block + text[insert_idx:], True

def _find_dynamic_region(text: str) -> Tuple[int, int, int, int]:
    """Return (region_start_line_idx, region_end_line_idx, body_start, body_end)."""
    start = text.find(_LIT_START)
    end   = text.find(_LIT_END)
    if start == -1 or end == -1 or end < start:
        raise RuntimeError("Dynamic region markers not found or malformed.")
    start_line_end = text.find("\n", start)
    if start_line_end == -1:
        start_line_end = len(text)
    body_start = start_line_end + 1
    body_end   = end
    return start, end, body_start, body_end

def _parse_blocks(region_text: str) -> Dict[str, str]:
    blocks: Dict[str, str] = {}
    for m in _BLOCK_RE.finditer(region_text):
        bid  = m.group("id")
        body = m.group("body").lstrip("\n").rstrip() + "\n"
        blocks[bid] = body
    return blocks

def _render_block(block_id: str, code: str) -> str:
    """Render a block with prolog/epilog to tag __MCP_ACTIVE_BLOCK_ID at import-time."""
    code_stripped = code.strip("\n") + "\n"
    prolog = f'__MCP_ACTIVE_BLOCK_ID = "{block_id}"\n'
    epilog = '\n__MCP_ACTIVE_BLOCK_ID = None\n'
    return f"{BLOCK_START} {block_id} START ---\n{prolog}{code_stripped}{epilog}{BLOCK_END}\n"

def _insert_or_replace_block(text: str, block_id: str, code: str, overwrite: bool) -> Tuple[str, bool, bool]:
    """Insert or replace a block inside the dynamic region."""
    text, _ = _ensure_dynamic_region_exists(text)
    _, _, body_start, body_end = _find_dynamic_region(text)
    region = text[body_start:body_end]
    blocks = _parse_blocks(region)

    created = False
    replaced = False
    rendered = _render_block(block_id, code)

    if block_id in blocks:
        if not overwrite:
            return text, False, False
        pattern = re.compile(
            rf"(#\s*---\s*BLOCK\s+{re.escape(block_id)}\s+START\s*---\s*\n).*?(#\s*---\s*END\s*---\s*\n)",
            re.DOTALL
        )
        new_region = pattern.sub(rendered, region, count=1)
        replaced = True
    else:
        if region and not region.endswith("\n"):
            region = region + "\n"
        new_region = region + rendered
        created = True

    new_text = text[:body_start] + new_region + text[body_end:]
    return new_text, created, replaced

# ---------------- Exec helpers ----------------
def _current_tool_names() -> List[str]:
    return list(_REGISTERED_TOOL_NAMES)

def _diff_tools(before: List[str], after: List[str]) -> List[str]:
    b, a = set(before), set(after)
    return [name for name in after if name not in b]

def _exec_dynamic_code(code: str, block_id: str) -> List[str]:
    """Compile+exec a block; record new tool names and block mapping."""
    compiled = compile(code, f"<dyn:{block_id}>", "exec")
    before = _current_tool_names()
    # Provide mcp and active block id so decorator can link tools to block
    g = {"mcp": mcp, "__name__": f"dyn_{block_id}", "__MCP_ACTIVE_BLOCK_ID": block_id}
    l: Dict[str, Any] = {}
    exec(compiled, g, l)
    after = _current_tool_names()
    added = _diff_tools(before, after)
    # Attach block mapping as extra safety (dedup)
    for name in added:
        meta = _TOOL_META.get(name, {})
        blist = list(dict.fromkeys(meta.get("blocks", []) + [block_id]))
        meta.update({"origin": meta.get("origin", "dynamic"), "blocks": blist})
        _TOOL_META[name] = meta
    _LOADED_BLOCK_IDS.add(block_id)
    return added

# ---------------- Self-heal core ----------------
def _scan_all_blocks(text: str) -> List[Tuple[int, int, str, str]]:
    """Return list of (start_idx, end_idx, id, body) for ALL blocks in file."""
    out = []
    for m in _BLOCK_RE.finditer(text):
        start, end = m.start(), m.end()
        bid  = m.group("id")
        body = m.group("body").lstrip("\n").rstrip() + "\n"
        out.append((start, end, bid, body))
    return out

def _remove_span(text: str, start: int, end: int) -> str:
    return text[:start] + text[end:]

@mcp.tool()
def repair_self_file(
    dry_run: bool = False,
    migrate_orphans: bool = True,
    prefer: str = "region"  # "region" keeps region copy if duplicate; "orphan" prefers orphan payload
) -> dict:
    """
    Self-heal this file:
      - Normalize marker lines to canonical.
      - Ensure dynamic region exists and is placed BEFORE the Entry section.
      - Find any orphan '# --- BLOCK ...' outside the region; migrate into region.
      - De-duplicate block ids (prefer='region' keeps region copy).
      - Fix lines that accidentally appended quotes to marker lines.
    Returns a summary and writes changes unless dry_run=True.
    """
    text = _read_self_text()
    original = text

    # 1) Ensure region exists and is before ENTRY
    text, changed1 = _ensure_dynamic_region_exists(text)

    # 2) Recompute region span
    reg_start, reg_end, body_start, body_end = _find_dynamic_region(text)
    changed2 = changed1

    # 3) Normalize marker lines again
    text, normed = _normalize_marker_lines(text)
    changed2 = changed2 or normed
    reg_start, reg_end, body_start, body_end = _find_dynamic_region(text)

    # 4) Move orphan blocks inside region (先删 orphan，再改动态区，避免索引错位)
    moved, ignored = [], []
    if migrate_orphans:
        all_blocks = _scan_all_blocks(text)
        region_blocks = _parse_blocks(text[body_start:body_end])
        for start, end, bid, body in reversed(all_blocks):
            inside = (start >= body_start and end <= body_end)
            if inside:
                continue
            if bid in region_blocks:
                if prefer == "orphan":
                    # 先删除 orphan
                    text = _remove_span(text, start, end)
                    # 再覆盖 region 版本
                    _, _, body_start, body_end = _find_dynamic_region(text)
                    pattern = re.compile(
                        rf"(#\s*---\s*BLOCK\s+{re.escape(bid)}\s+START\s*---\s*\n).*?(#\s*---\s*END\s*---\s*\n)",
                        re.DOTALL
                    )
                    region_text = text[body_start:body_end]
                    region_text = pattern.sub(_render_block(bid, body), region_text, count=1)
                    text = text[:body_start] + region_text + text[body_end:]
                    changed2 = True
                    moved.append(bid)
                else:
                    # 保留 region 版本，丢弃 orphan
                    text = _remove_span(text, start, end)
                    changed2 = True
                    ignored.append(bid)
            else:
                # 先删除 orphan，再把内容附加进 region
                text = _remove_span(text, start, end)
                _, _, body_start, body_end = _find_dynamic_region(text)
                region_text = text[body_start:body_end]
                if region_text and not region_text.endswith("\n"):
                    region_text += "\n"
                region_text += _render_block(bid, body)
                text = text[:body_start] + region_text + text[body_end:]
                changed2 = True
                moved.append(bid)
        # Recompute region span after edits
        reg_start, reg_end, body_start, body_end = _find_dynamic_region(text)

    # 5) Clean accidental trailing quotes on marker lines
    text = re.sub(r'(# === DYNAMIC TOOLS (?:START|END) ===)"\s*$', r'\1', text, flags=re.MULTILINE)

    summary = {
        "changed": changed2 or (text != original),
        "moved_orphans": sorted(list(set(moved))),
        "ignored_duplicates": sorted(list(set(ignored))),
    }
    if not dry_run and summary["changed"]:
        _write_self_text(text)
        summary["written"] = True
    else:
        summary["written"] = False
    return summary

# ---------------- Block management tools ----------------
@mcp.tool()
def add_tool_block(
    code: str,
    block_id: Optional[str] = None,
    overwrite: bool = False,
    activate: bool = True,
) -> dict:
    """
    Add a new dynamic @mcp.tool() block into THIS file and (optionally) activate immediately.
    """
    code = code.strip()
    if not code:
        return {"error": "Empty code."}
    if block_id is None or not re.match(r"^[A-Za-z0-9_\-]+$", block_id or ""):
        block_id = uuid.uuid4().hex[:8]

    # preflight
    try:
        compile(code, f"<dyn:{block_id}>", "exec")
    except Exception as e:
        return {"block_id": block_id, "persisted": False, "activated": False,
                "error": f"Compile error: {e!r}"}

    # activate
    new_tools: List[str] = []
    if activate:
        try:
            new_tools = _exec_dynamic_code(code, block_id)
        except Exception as e:
            return {"block_id": block_id, "persisted": False, "activated": False,
                    "error": f"Runtime error during activation: {e!r}"}

    # persist
    try:
        text = _read_self_text()
        new_text, created, replaced = _insert_or_replace_block(text, block_id, code, overwrite=overwrite)
        persisted = False
        if created or replaced:
            _write_self_text(new_text)
            persisted = True

        # Notify clients that tool list changed (if any new tools or replaced)
        if activate and new_tools:
            _notify_tools_changed_safe()
        elif (created or replaced):
            _notify_tools_changed_safe()

        return {
            "block_id": block_id,
            "persisted": persisted,
            "created": created,
            "replaced": replaced,
            "activated": bool(activate),
            "new_tools": new_tools,
            "notes": "OK" if (persisted or activate) else "Block existed; skipped (no overwrite).",
        }
    except Exception as e:
        return {"block_id": block_id, "persisted": False, "activated": bool(activate),
                "new_tools": new_tools, "error": f"Persist error: {e!r}"}

@mcp.tool()
def list_tool_blocks() -> dict:
    """List persisted blocks and their tools (best-effort)."""
    text, _ = _ensure_dynamic_region_exists(_read_self_text())
    _, _, body_start, body_end = _find_dynamic_region(text)
    region = text[body_start:body_end]
    blocks = _parse_blocks(region)
    return {
        "blocks": sorted(blocks.keys()),
        "by_block_tools": {bid: [name for name, meta in _TOOL_META.items() if bid in meta.get("blocks", [])]
                           for bid in blocks.keys()},
        "loaded": sorted(list(_LOADED_BLOCK_IDS)),
    }

@mcp.tool()
def show_tool_block(block_id: str) -> str:
    """Show the source code of a specific persisted block."""
    text, _ = _ensure_dynamic_region_exists(_read_self_text())
    _, _, body_start, body_end = _find_dynamic_region(text)
    region = text[body_start:body_end]
    blocks = _parse_blocks(region)
    if block_id not in blocks:
        return f"[not found] block_id={block_id}"
    return blocks[block_id]

@mcp.tool()
def remove_tool_block(block_id: str) -> dict:
    """
    Remove a block from the FILE. Tools already registered in THIS PROCESS remain until restart.
    """
    text, _ = _ensure_dynamic_region_exists(_read_self_text())
    _, _, body_start, body_end = _find_dynamic_region(text)
    region = text[body_start:body_end]
    pattern = re.compile(
        rf"(#\s*---\s*BLOCK\s+{re.escape(block_id)}\s+START\s*---\s*\n).*?(#\s*---\s*END\s*---\s*\n)",
        re.DOTALL
    )
    if not pattern.search(region) and block_id not in _LOADED_BLOCK_IDS:
        return {"ok": False, "error": f"Block not found: {block_id}"}
    new_region = pattern.sub("", region, count=1)
    new_text = text[:body_start] + new_region + text[body_end:]
    _write_self_text(new_text)

    # Notify removal
    _notify_tools_changed_safe()
    return {"ok": True, "block_id": block_id,
            "notes": "Removed from file. Restart to fully drop runtime registrations of this block."}

@mcp.tool()
def list_registered_tools() -> dict:
    """Return all registered tool names with docs/signatures if recorded."""
    out = {}
    for name in _current_tool_names():
        meta = _TOOL_META.get(name, {})
        out[name] = {
            "signature": meta.get("signature", "(...)"),
            "doc": meta.get("doc", ""),
            "origin": meta.get("origin", "builtin"),
            "blocks": meta.get("blocks", []),
        }
    return out

@mcp.tool()
def describe_tool(name: str) -> dict:
    """Describe a tool's signature and docstring (if recorded)."""
    meta = _TOOL_META.get(name)
    if not meta:
        return {"ok": False, "error": f"Tool not found or metadata missing: {name}"}
    return {"ok": True, "name": name, "signature": meta.get("signature", "(...)"),
            "doc": meta.get("doc", ""), "origin": meta.get("origin", "unknown"),
            "blocks": meta.get("blocks", [])}

@mcp.tool()
def usage_example_for_tool(name: str) -> dict:
    """Generate a minimal example args payload from a tool's signature (best-effort)."""
    meta = _TOOL_META.get(name)
    if not meta:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    sig = meta.get("signature", "(...)")
    example = {}
    try:
        inside = sig.strip()
        if inside.startswith("(") and inside.endswith(")"):
            inside = inside[1:-1]
        parts = [p.strip() for p in inside.split(",")] if inside else []
        for p in parts:
            if not p:
                continue
            name_part = p.split(":", 1)[0].split("=", 1)[0].strip()
            if name_part:
                example[name_part] = f"<{name_part}>"
    except Exception:
        pass
    return {"ok": True, "tool": name, "signature": sig, "example_args": example, "doc": meta.get("doc", "")}

# ---------------- Lint & Safe Add tools ----------------
@mcp.tool()
def lint_block_code(code: str) -> dict:
    """
    静态检查动态块的常见致命问题：
      - 重新实例化 FastMCP / 重绑定 mcp
      - 把定义写进 if __name__ == "__main__"
      - 没有 @mcp.tool() 装饰器
    """
    import re as _re
    issues: List[str] = []
    # 1) 重新创建/重绑定 mcp
    if _re.search(r'^\s*from\s+mcp\.server\.fastmcp\s+import\s+FastMCP', code, _re.M):
        issues.append("Imports FastMCP (不要在动态块创建新的 MCP 实例)")
    if _re.search(r'^\s*import\s+mcp\.server\.fastmcp', code, _re.M):
        issues.append("Imports mcp.server.fastmcp (高风险会新建 MCP)")
    if _re.search(r'^\s*mcp\s*=', code, _re.M):
        issues.append("Rebinds variable 'mcp' (会注册到错误实例)")
    # 2) main guard
    if _re.search(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:', code):
        issues.append("Has if __name__ == '__main__' guard (动态执行时不会触发)")
    # 3) 是否至少有一个装饰器
    if "@mcp.tool" not in code:
        issues.append("No '@mcp.tool()' decorator found")
    return {"ok": True, "issues": issues}

@mcp.tool()
def diagnose_block(block_id: str) -> dict:
    """读取已持久化块并做 lint，帮助定位为何工具没注册上。"""
    src = show_tool_block(block_id)
    if src.startswith("[not found]"):
        return {"ok": False, "error": f"block not found: {block_id}"}
    res = lint_block_code(src)
    return {"ok": True, "block_id": block_id, "issues": res.get("issues", []), "length": len(src)}

@mcp.tool()
def safe_add_tool_block(
    code: str,
    block_id: Optional[str] = None,
    overwrite: bool = False,
    activate: bool = True,
    strict: bool = True
) -> dict:
    """
    带静态体检的安全注入。strict=True 时若发现致命问题将拒绝注入并返回 issues。
    """
    lint = lint_block_code(code)
    issues = lint.get("issues", [])
    fatal_markers = ("FastMCP", "Rebinds variable 'mcp'", "if __name__ == '__main__'", "No '@mcp.tool()'")
    fatal = any(any(tok in m for tok in fatal_markers) for m in issues)
    if strict and fatal:
        return {"ok": False, "error": "lint_failed", "issues": issues}
    res = add_tool_block(code=code, block_id=block_id, overwrite=overwrite, activate=activate)  # reuse original
    res.update({"lint_issues": issues, "ok": "error" not in res})
    return res

# ---------------- One-shot: create a uniquely-named Newton cooling tool ----------------
@mcp.tool()
def create_newton_cooling_tool(suffix: Optional[str] = None,
                               block_id: Optional[str] = None,
                               activate: bool = True) -> dict:
    """
    自动生成一个“唯一函数名”的牛顿冷却工具并注入。
    目的：绕过客户端缓存/重名冲突，验证注册链路。
    """
    suf = (suffix or uuid.uuid4().hex[:8]).lower()
    fname = f"run_thermo_simulation_{suf}"
    bid = block_id or f"newton_cooling_{suf}"
    code = f'''
import math, os
from pathlib import Path

@mcp.tool()
def {fname}(
    k: float,
    T0: float,
    Tenv: float,
    t1: float = 60.0,
    dt: float = 0.5,
    image_filename: str = "newtons_cooling_plot.png",
    return_csv: bool = False
) -> dict:
    """
    Newton's cooling: dT/dt = -k*(T - Tenv).
    Integrates with RK4; plots to PNG if matplotlib is available, else optional CSV.
    """
    if dt <= 0: raise ValueError("dt must be > 0")
    if t1 <= 0: raise ValueError("t1 must be > 0")

    times = [0.0]; temps = [float(T0)]
    t = 0.0; T = float(T0)
    nsteps = int(max(1, round(t1 / dt)))
    h = t1 / nsteps

    def f(temp, _t): return -k * (temp - Tenv)

    for _ in range(nsteps):
        k1 = f(T, t)
        k2 = f(T + 0.5*h*k1, t + 0.5*h)
        k3 = f(T + 0.5*h*k2, t + 0.5*h)
        k4 = f(T + h*k3, t + h)
        T = T + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        times.append(t); temps.append(T)

    out_dir = Path(os.environ.get("MCP_LATEX_OUTPUT_DIR", ".")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / image_filename

    csv_path = None
    if return_csv:
        csv_path = out_dir / (Path(image_filename).stem + "_data.csv")
        with open(csv_path, "w", encoding="utf-8") as fp:
            fp.write("t,T\\n")
            for tt, temp in zip(times, temps):
                fp.write(f"{{tt}},{{temp}}\\n")

    plotted = False
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(times, temps, label="T(t)")
        plt.title("Newton's Cooling")
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.legend()
        fig.savefig(str(image_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        plotted = True
    except Exception:
        plotted = False

    return {{
        "ok": True,
        "tool_name": "{fname}",
        "plotted": plotted,
        "image_path": str(image_path) if plotted else None,
        "csv_path": str(csv_path) if csv_path else None,
        "n_samples": len(times),
        "sample_head": {{"times": times[:5], "temps": temps[:5]}},
        "params": {{"k": k, "T0": T0, "Tenv": Tenv, "t1": t1, "dt": dt}}
    }}
'''
    res = safe_add_tool_block(code=code, block_id=bid, overwrite=False, activate=activate, strict=True)  # type: ignore
    res.update({"generated_tool": fname, "block_id": bid})
    return res

# ---------------- Persisted-block (re)loading ----------------
def _load_all_blocks_impl(force: bool = False) -> List[str]:
    """Exec all persisted blocks that aren't loaded yet (or all if force=True)."""
    text, _ = _ensure_dynamic_region_exists(_read_self_text())
    _, _, body_start, body_end = _find_dynamic_region(text)
    blocks = _parse_blocks(text[body_start:body_end])
    loaded_now: List[str] = []
    for bid, body in blocks.items():
        if not force and bid in _LOADED_BLOCK_IDS:
            continue
        try:
            _exec_dynamic_code(body, bid)
            loaded_now.append(bid)
        except Exception as e:
            log.exception("Failed loading block %s: %r", bid, e)
    return loaded_now

@mcp.tool()
def reload_all_blocks(force: bool = False) -> dict:
    """Load persisted blocks into runtime. If force=True, re-exec all blocks."""
    loaded = _load_all_blocks_impl(force=force)
    if loaded:
        _notify_tools_changed_safe()
    return {"ok": True, "loaded_block_ids": loaded, "already_loaded": sorted(list(_LOADED_BLOCK_IDS - set(loaded)))}

# ---------------- Built-in health ----------------
@mcp.tool()
def healthcheck() -> str:
    """Return 'ok' for liveness probing."""
    return "ok"

@mcp.tool()
def announce_tools_changed() -> dict:
    """Manually trigger 'tools list changed' notification to clients."""
    ok = _notify_tools_changed_safe()
    return {"ok": ok}

# ---------------- Offline template library ----------------
_TEMPLATE_THERMO_ODE_RK4 = r'''
import math, ast
from typing import List, Optional
import os
from pathlib import Path

@mcp.tool()
def thermo_integrate_ode(
    variables: List[str],
    rhs: dict,
    y0: dict,
    t0: float,
    t1: float,
    dt: float,
    params: Optional[dict] = None,
    save_every: int = 1,
    max_steps: int = 20000,
) -> dict:
    """
    通用 ODE / 热力学微分方程四阶 RK 积分器（仅标准库）。
    - variables: 变量名列表，如 ["T","P","V"]。
    - rhs: 每个变量的右端表达式（字符串），可用变量名、t、params 中常量；允许函数
           sin, cos, tan, sinh, cosh, tanh, exp, log, sqrt, pow, abs, min, max；常数 pi, e。
           例如 Newton 冷却: {"T":"-k*(T - Tenv)"}。
    - y0: 初值 dict；t0, t1, dt: 自变量区间与步长；params: 常量字典；save_every: 保存间隔。
    返回:
      {"variables":[...], "steps":N, "times":[...], "states":[{var:value,...}, ...]}
    注意：刚性方程请选小步长；单位自洽。
    """
    try:
        import matplotlib.pyplot as plt  # optional for plotting
    except Exception:
        plt = None

    if params is None:
        params = {}

    SAFE_FUNCS = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
        "exp": math.exp, "log": math.log, "sqrt": math.sqrt,
        "pow": pow, "abs": abs, "min": min, "max": max,
    }
    CONSTS = {"pi": math.pi, "e": math.e}

    def safe_eval(expr: str, state: dict, tval: float) -> float:
        names = {}
        names.update(CONSTS); names.update(params); names.update(state); names.update({"t": tval})
        import ast as _ast
        tree = _ast.parse(expr, mode="eval")
        def ev(node):
            if isinstance(node, _ast.Expression): return ev(node.body)
            if isinstance(node, _ast.Constant):
                if isinstance(node.value, (int, float)): return float(node.value)
                raise ValueError("only numeric constants allowed")
            if isinstance(node, _ast.Name):
                if node.id in names: return float(names[node.id])
                raise ValueError(f"unknown name: {node.id}")
            if isinstance(node, _ast.BinOp) and isinstance(node.op, (_ast.Add,_ast.Sub,_ast.Mult,_ast.Div,_ast.Pow,_ast.Mod)):
                l = ev(node.left); r = ev(node.right)
                if isinstance(node.op, _ast.Add): return l+r
                if isinstance(node.op, _ast.Sub): return l-r
                if isinstance(node.op, _ast.Mult): return l*r
                if isinstance(node.op, _ast.Div): return l/r
                if isinstance(node.op, _ast.Pow): return l**r
                if isinstance(node.op, _ast.Mod): return l%r
            if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
                v = ev(node.operand); return v if isinstance(node.op,_ast.UAdd) else -v
            if isinstance(node, _ast.Call) and isinstance(node.func, _ast.Name):
                fname = node.func.id
                if fname not in SAFE_FUNCS: raise ValueError(f"function not allowed: {fname}")
                if node.keywords: raise ValueError("keyword arguments not allowed")
                args = [ev(a) for a in node.args]
                return float(SAFE_FUNCS[fname](*args))
            raise ValueError("disallowed expression")
        return float(ev(tree))

    # 封装 RHS
    variables = list(variables)
    for v in variables:
        if v not in rhs: raise ValueError(f"rhs missing: {v}")
        if v not in y0:  raise ValueError(f"y0 missing: {v}")
    f = {v: (lambda e: (lambda st, tt, _e=e: safe_eval(_e, st, tt)))(str(rhs[v])) for v in variables}

    # RK4
    direction = 1.0 if t1 >= t0 else -1.0
    h0 = abs(float(dt)) * direction
    if h0 == 0.0: raise ValueError("dt must be non-zero")
    t = float(t0); state = {v: float(y0[v]) for v in variables}
    times = [t]; traj = [dict(state)]; steps = 0
    while True:
        if steps >= max_steps: break
        if (direction > 0 and t >= t1) or (direction < 0 and t <= t1): break
        h = h0
        rem = (t1 - t)
        if (direction > 0 and t + h > t1) or (direction < 0 and t + h < t1): h = rem
        k1 = {v: f[v](state, t) for v in variables}
        s2 = {v: state[v] + 0.5*h*k1[v] for v in variables}
        k2 = {v: f[v](s2, t + 0.5*h) for v in variables}
        s3 = {v: state[v] + 0.5*h*k2[v] for v in variables}
        k3 = {v: f[v](s3, t + 0.5*h) for v in variables}
        s4 = {v: state[v] + h*k3[v] for v in variables}
        k4 = {v: f[v](s4, t + h) for v in variables}
        state = {v: state[v] + (h/6.0)*(k1[v] + 2*k2[v] + 2*k3[v] + k4[v]) for v in variables}
        t = t + h; steps += 1
        if steps % int(max(1, save_every)) == 0:
            times.append(t); traj.append(dict(state))
    if times[-1] != t:
        times.append(t); traj.append(dict(state))

    # 可选画图
    out_dir = Path(os.environ.get("MCP_LATEX_OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_dir / "thermo_ode_trajectory.png"
    if 'plt' in locals() and plt is not None:
        fig = plt.figure()
        for v in variables:
            plt.plot(times, [s[v] for s in traj], label=v)
        plt.title("ODE Trajectory"); plt.xlabel("t"); plt.ylabel("value"); plt.legend()
        fig.savefig(str(img_path), dpi=200, bbox_inches="tight"); plt.close(fig)
    return {"variables": variables, "steps": steps, "times": times, "states": traj, "image_path": str(img_path)}
'''

_TEMPLATE_LATEX_IO_BASIC = r'''
from pathlib import Path
import os
from typing import List, Optional

@mcp.tool()
def write_bib_and_latex(
    title: str,
    author: str,
    abstract: str,
    figure_path: str = "newtons_cooling_plot.png",
    bib_entries: Optional[List[str]] = None,
    tex_filename: str = "main.tex",
    bib_filename: str = "refs.bib"
) -> dict:
    """
    写入 refs.bib 与 main.tex（插入图与参考文献）。路径基于 MCP_LATEX_OUTPUT_DIR 或当前目录。
    """
    out_dir = Path(os.environ.get("MCP_LATEX_OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    bib_path = out_dir / bib_filename
    tex_path = out_dir / tex_filename
    bib_text = ""
    if bib_entries:
        bib_text = "\n\n".join([e.strip() for e in bib_entries if e.strip()])
    else:
        bib_text = """@article{NewtonCooling,
  title   = {On cooling law},
  author  = {Newton, Isaac},
  journal = {Philosophical Transactions},
  year    = {1701}
}"""
    bib_path.write_text(bib_text, encoding="utf-8")
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage{{graphicx}}
\usepackage{{authblk}}
\usepackage{{cite}}
\title{{{title}}}
\author{{{author}}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{abstract}
\end{{abstract}}
\begin{{figure}}[h]
  \centering
  \includegraphics[width=0.75\linewidth]{{{figure_path}}}
  \caption{{Evolution plot.}}
\end{{figure}}
\bibliographystyle{{unsrt}}
\bibliography{{{Path(bib_filename).stem}}}
\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")
    return {"ok": True, "tex": str(tex_path), "bib": str(bib_path)}
'''

_TEMPLATE_INSPIRE_MIN = r'''
import httpx, json, logging
from typing import Dict, Any, List
from mcp.types import TextContent

log = logging.getLogger("inspire-tools")
INSPIRE_BASE = "https://inspirehep.net/api"
DEFAULT_HEADERS = {"User-Agent": "inspire-mcp/mini"}
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
LIT_FIELDS = ",".join([
    "control_number","titles.title","authors.full_name","arxiv_eprints.value",
    "earliest_date","citation_count","publication_info.year"
])

def _extract_min(md: Dict[str, Any]) -> Dict[str, Any]:
    title = (md.get("titles") or [{}])[0].get("title")
    arxiv_id = (md.get("arxiv_eprints") or [{}])[0].get("value") if md.get("arxiv_eprints") else None
    pubinfo = md.get("publication_info") or []
    year = pubinfo[0].get("year") if pubinfo and isinstance(pubinfo, list) else None
    if (not year) and isinstance(md.get("earliest_date"), str) and len(md["earliest_date"])>=4:
        year = md["earliest_date"][:4]
    return {
        "inspire_id": md.get("control_number"),
        "title": title,
        "authors": [a.get("full_name") for a in (md.get("authors") or []) if a.get("full_name")] or None,
        "year": year,
        "arxiv_id": arxiv_id,
        "arxiv_abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "inspire_url": f"https://inspirehep.net/literature/{md.get('control_number')}" if md.get("control_number") else None,
        "citation_count": md.get("citation_count"),
    }

def _build_topic_query(topic: str, exact: bool, start_year: int, end_year: int) -> str:
    phrase = topic.strip().replace('"','\\"')
    token = f'"{phrase}"' if exact else phrase
    subqs = [f"abstracts.value:{token}", f"t:{token}", f"k:{token}"]
    date = ""
    if start_year>0 and end_year>0 and end_year>=start_year:
        date = f" and date:{start_year}->{end_year}"
    return "(" + " or ".join(subqs) + ")" + date

@mcp.tool()
async def inspire_search_topic(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    exact_phrase: bool = True,
    as_json: bool = True,
) -> TextContent:
    """
    Minimal INSPIRE-HEP topic search (server-side relevance sort).
    """
    q = _build_topic_query(topic, exact_phrase, start_year, end_year)
    params = {"q": q, "size": max(1, min(size, 50)), "fields": LIT_FIELDS}
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS) as client:
        try:
            resp = await client.get(f"{INSPIRE_BASE}/literature", params=params)
            resp.raise_for_status()
            data = resp.json()
            hits = (data.get("hits") or {}).get("hits") or []
            results = [_extract_min(h.get("metadata") or {}) for h in hits]
            payload = {"query_used": q, "count": len(results), "results": results}
            return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))
        except httpx.HTTPError as e:
            return TextContent(type="text", text=f"HTTP error: {e}")
        except Exception as e:
            return TextContent(type="text", text=f"Error: {type(e).__name__}: {e}")
'''

_TEMPLATES: Dict[str, str] = {
    "thermo_ode_rk4": _TEMPLATE_THERMO_ODE_RK4,
    "latex_io_basic": _TEMPLATE_LATEX_IO_BASIC,
    "inspire_tools_min": _TEMPLATE_INSPIRE_MIN,  # optional; requires httpx
}

@mcp.tool()
def list_templates() -> List[str]:
    """List available built-in templates."""
    return sorted(_TEMPLATES.keys())

@mcp.tool()
def get_template(name: str) -> dict:
    """Return the source of a template for review."""
    if name not in _TEMPLATES:
        return {"ok": False, "error": f"Unknown template: {name}"}
    return {"ok": True, "name": name, "code": _TEMPLATES[name]}

@mcp.tool()
def install_template(
    name: str,
    block_id: Optional[str] = None,
    overwrite: bool = False,
    activate: bool = True
) -> dict:
    """Install a built-in template as a dynamic block (persist into THIS file)."""
    if name not in _TEMPLATES:
        return {"ok": False, "error": f"Unknown template: {name}"}
    code = _TEMPLATES[name]
    if block_id is None:
        block_id = f"{name}-{uuid.uuid4().hex[:6]}"
    res = add_tool_block(code=code, block_id=block_id, overwrite=overwrite, activate=activate)  # type: ignore
    res.update({"ok": True, "template": name})
    return res

# ---------------- Resources ----------------
@mcp.resource("self://source")
def show_self_source() -> str:
    """Return the entire source code of this file (for review)."""
    try:
        return _read_self_text()
    except Exception as e:
        return f"[error reading self] {e!r}"

# === DYNAMIC TOOLS START ===
# (dynamic @mcp.tool() blocks will be inserted here)

# --- BLOCK thermo_solver START ---
__MCP_ACTIVE_BLOCK_ID = "thermo_solver"

from typing import Optional

@mcp.tool()
def solve_thermo_ode_with_mathematica(
    ode_system,
    initial_conditions,
    t_range,
    plot_options: Optional[dict] = None
):
    """
    Solves a system of ODEs using Mathematica (wolframscript) and optionally generates a plot.

    Args:
        ode_system: A list of strings, each an ODE in Mathematica format.
                    Example: ["T'[t] == -k (T[t] - T_env)"]
        initial_conditions: A list of strings, initial conditions in Mathematica format.
                            Example: ["T[0] == 100"]
        t_range: A dict like {"start": 0, "end": 10}
        plot_options: Optional dict for plotting:
            {
              "plot_title": "Newton's Law of Cooling",
              "x_label": "Time (s)",
              "y_label": "Temperature (C)",
              "image_name": "newton_cooling.png"
            }
    Returns:
        {"solution": <stdout string from wolframscript>, "plot_path": <str|None>}
    """
    import os
    from mathematica_check import execute_mathematica  # local module that wraps wolframscript

    code = f"""
    solution = NDSolve[
        {{ {", ".join(ode_system)}, {", ".join(initial_conditions)} }},
        T,
        {{t, {t_range['start']}, {t_range['end']}}}
    ];
    """

    image_path = None
    if plot_options:
        output_dir = os.path.join(os.getcwd(), "thermo_output")
        os.makedirs(output_dir, exist_ok=True)
        image_path = os.path.join(output_dir, plot_options.get('image_name', 'plot.png')).replace("\\\\", "/")

        code += f"""
        plot = Plot[T[t] /. solution, {{t, {t_range['start']}, {t_range['end']}}},
            PlotLabel -> "{plot_options.get('plot_title', '')}",
            AxesLabel -> {{"{plot_options.get('x_label','t')}", "{plot_options.get('y_label','T')}" }},
            PlotStyle -> Thick
        ];
        Export["{image_path}", plot];
        """

    result = execute_mathematica(code)  # may take time; consider timeout settings in that server
    ret = {"solution": result}
    if image_path:
        ret["plot_path"] = image_path
    return ret

__MCP_ACTIVE_BLOCK_ID = None
# --- END ---

# === DYNAMIC TOOLS END ===

# ---------------- Entry ----------------
if __name__ == "__main__":
    # Auto-self-heal (repair markers/region, migrate orphans), then load persisted blocks
    try:
        rep = repair_self_file(dry_run=False, migrate_orphans=True, prefer="region")  # type: ignore
        log.info("self-heal summary: %s", rep)
    except Exception:
        log.exception("self-heal failed on startup")

    try:
        loaded = _load_all_blocks_impl(force=False)
        if loaded:
            _notify_tools_changed_safe()
        log.info("loaded blocks at startup: %s", loaded)
    except Exception:
        log.exception("loading blocks failed on startup")

    mcp.run("stdio")
