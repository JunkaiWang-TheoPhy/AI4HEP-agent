# server.py — INSPIRE-HEP MCP server
# (semantic topic search + hybrid multi-channel retrieval + core-token AND + classic fallback + reference expansion + bibtex verification)

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import logging
import httpx
import json
import re
import math
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Set
from urllib.parse import urlparse

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("inspire-mcp-semantic")

# ------------------------------
# FastMCP
# ------------------------------
mcp = FastMCP()

# ------------------------------
# HTTP client settings
# ------------------------------
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=12.0, write=5.0, pool=5.0)
DEFAULT_HEADERS = {"User-Agent": "inspire-mcp/semantic-1.1"}
HTTP_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)

INSPIRE_BASE = "https://inspirehep.net/api"

# 基础字段（权威）
LIT_FIELDS = ",".join([
    "control_number",
    "titles.title",
    "authors.full_name",
    "arxiv_eprints.value",
    "earliest_date",
    "citation_count",
    "citation_count_without_self_citations",
    "publication_info.year",
])

# 扩展字段：摘要 + references
LIT_FIELDS_WITH_ABS = LIT_FIELDS + ",abstracts.value"
REF_ONLY_FIELDS = "references.record.$ref,references.reference.title,references.reference.arxiv_eprint,references.reference.dois.value"

# ------------------------------
# hep-th 常见同义词/概念（含中文桥接）
# ------------------------------
SYNONYM_MAP: Dict[str, List[str]] = {
    # 经典主题
    "holographic entanglement entropy": [
        "holographic entanglement entropy", "entanglement entropy",
        "ryu takayanagi", "ryu–takayanagi", "rt formula", "hrt", "hubeny rangamani takayanagi",
        "holographic entropy", "minimal surface", "geodesic length",
        "entanglement wedge", "islands", "cosmic brane", "quantum extremal surface", "qes",
        "ads/cft", "adscft", "gauge/gravity"
    ],
    "entanglement entropy": [
        "entanglement entropy", "von neumann entropy", "renyi entropy", "rényi entropy",
        "mutual information", "entanglement negativity"
    ],
    "ads/cft": [
        "ads/cft", "adscft", "ads cft", "ads3/cft2", "gauge/gravity duality", "holography", "holographic"
    ],
    "black hole": [
        "black hole", "black holes", "bh", "kerr", "schwarzschild", "bekenstein", "hawking",
        "bekenstein hawking", "event horizon", "btz"
    ],
    "black hole entropy": [
        "black hole entropy", "bekenstein-hawking entropy", "bh entropy", "gravitational entropy"
    ],
    "quantum gravity": [
        "quantum gravity", "qg", "quantum spacetime", "non-perturbative gravity"
    ],
    "string theory": [
        "string theory", "superstring", "m-theory", "heterotic", "type ii", "type iia", "type iib"
    ],
    "conformal field theory": [
        "conformal field theory", "cft", "2d cft", "two-dimensional cft", "conformal symmetry"
    ],
    "supersymmetry": [
        "supersymmetry", "susy", "n=4", "n=2", "superconformal"
    ],
    "scattering amplitude": [
        "scattering amplitude", "amplitude", "on-shell", "twistor", "bcfw"
    ],
    # —— 中文桥接 ——（将中文键映射到英文检索词，支持中文输入）
    "全息纠缠熵": [
        "holographic entanglement entropy", "entanglement entropy",
        "ryu takayanagi", "rt formula", "hrt", "entanglement wedge", "minimal surface", "ads/cft"
    ],
    "量子极值曲面": [
        "quantum extremal surface", "qes", "entanglement wedge", "island", "islands", "cosmic brane"
    ],
    "瑞-高柳": [
        "ryu takayanagi", "rt formula", "holographic entanglement entropy"
    ],
}

# ------------------------------
# 文本与术语处理（轻量形态变体 + 语义匹配）
# ------------------------------
_non_alnum_re = re.compile(r"[^a-z0-9]+")

def _normalize_token(s: str) -> str:
    return _non_alnum_re.sub(" ", s.lower()).strip()

def _tokenize(s: Optional[str]) -> List[str]:
    if not s:
        return []
    base = _normalize_token(s)
    toks = [t for t in base.split() if t]
    return toks

def _expand_variants(term: str) -> Set[str]:
    """朴素派生：s/es/ies、ic/ical、去空格/连字符/斜线、连写等"""
    t = term.strip().lower()
    out: Set[str] = set([t])
    out.add(t.replace(" ", ""))
    out.add(t.replace("-", ""))
    out.add(t.replace("/", ""))
    out.add(t.replace(" ", "-"))
    out.add(t.replace(" ", "/"))
    if t.endswith("ic"):
        out.add(t + "al")
    if t.endswith("ical"):
        out.add(t[:-2])
    if not t.endswith("s"):
        out.add(t + "s")
    if t.endswith("y"):
        out.add(t[:-1] + "ies")
    if t.endswith("is"):
        out.add(t[:-2] + "es")
    return {v for v in out if len(v) >= 2}

def _concept_groups_from_topic(topic: str) -> List[List[str]]:
    """
    1) 先匹配内置 SYNONYM_MAP 的 key（长 key 优先，支持中文键 → 英文同义词）
    2) 剩余词作为补充组（>=3 字母）
    3) 每组做形态扩展
    """
    t_low = topic.lower()
    picked_keys: List[str] = []
    for k in sorted(SYNONYM_MAP.keys(), key=len, reverse=True):
        if k in t_low:
            picked_keys.append(k)

    groups: List[List[str]] = []
    for k in picked_keys:
        base = [k] + SYNONYM_MAP.get(k, [])
        groups.append(list(dict.fromkeys([b.lower() for b in base])))

    covered = set()
    for k in picked_keys:
        covered |= set(_tokenize(k))
    residual = [w for w in _tokenize(t_low) if w not in covered and len(w) >= 3]
    if residual:
        groups.append(residual)

    if not groups:
        groups = [[topic.lower()]]

    groups_expanded: List[List[str]] = []
    for g in groups:
        exp: Set[str] = set()
        for term in g:
            for v in _expand_variants(term):
                exp.add(v)
        groups_expanded.append(sorted(exp))
    return groups_expanded

def _top_core_tokens(variants: List[str], max_tokens: int = 3) -> List[str]:
    """
    从一个概念组的变体里抽取“核心词”（按词频优先，其次长度优先），用于 AND 通道。
    避免选到 QES/Islands 等后起概念，从而误伤 0603001 这类早期经典。
    """
    freq: Dict[str, int] = {}
    for v in variants:
        for tok in _tokenize(v):
            if len(tok) < 3:
                continue
            freq[tok] = freq.get(tok, 0) + 1
    core = sorted(freq.keys(), key=lambda t: (-freq[t], -len(t), t))
    return core[:max_tokens] if core else []

def _text_semantic_score(title: Optional[str], abstract: Optional[str], topic: str) -> float:
    """
    语义得分（0~1）：基于概念组覆盖 + 标题权重 + 频次。
    - 标题权重 2.5，摘要权重 1.0。
    """
    groups = _concept_groups_from_topic(topic)
    if not groups:
        return 0.0
    toks_title = set(_tokenize(title))
    toks_abs = set(_tokenize(abstract))
    if not toks_title and not toks_abs:
        return 0.0

    def group_hit(tokens: Set[str], variants: List[str]) -> float:
        max_h = 0.0
        for v in variants:
            parts = v.split()
            if len(parts) >= 2:
                if set(parts).issubset(tokens):
                    max_h = max(max_h, 1.0)
            else:
                if parts[0] in tokens:
                    max_h = max(max_h, 0.8)
        return max_h

    w_t, w_a = 2.5, 1.0
    hits_t = sum(group_hit(toks_title, g) for g in groups)
    hits_a = sum(group_hit(toks_abs, g) for g in groups)
    cov = (w_t * hits_t + w_a * hits_a) / (w_t + w_a)
    cov /= max(1.0, float(len(groups)))
    covered_groups = 0
    for g in groups:
        if group_hit(toks_title, g) > 0 or group_hit(toks_abs, g) > 0:
            covered_groups += 1
    freq = min(1.0, covered_groups / max(1.0, len(groups)))
    score = 0.7 * cov + 0.3 * freq
    return max(0.0, min(1.0, score))

# ------------------------------
# INSPIRE helpers
# ------------------------------
def _extract_record_min(md: Dict[str, Any]) -> Dict[str, Any]:
    title = (md.get("titles") or [{}])[0].get("title")
    arxiv_id = (md.get("arxiv_eprints") or [{}])[0].get("value") if md.get("arxiv_eprints") else None
    pubinfo = md.get("publication_info") or []
    year = pubinfo[0].get("year") if pubinfo and isinstance(pubinfo, list) else None
    if (not year) and isinstance(md.get("earliest_date"), str) and len(md["earliest_date"]) >= 4:
        year = md["earliest_date"][:4]
    return {
        "inspire_id": md.get("control_number") or None,
        "title": title,
        "authors": [a.get("full_name") for a in (md.get("authors") or []) if a.get("full_name")] or None,
        "earliest_date": md.get("earliest_date"),
        "year": year,
        "citation_count": md.get("citation_count"),
        "citation_count_wo_self": md.get("citation_count_without_self_citations"),
        "arxiv_id": arxiv_id,
        "arxiv_abs_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
        "inspire_url": f"https://inspirehep.net/literature/{md.get('control_number')}" if md.get("control_number") else None,
        "abstract": ((md.get("abstracts") or [{}])[0].get("value") if md.get("abstracts") else None),
    }

def _norm_key(rec: Dict[str, Any]) -> str:
    if rec.get("arxiv_id"):
        return f"arxiv:{rec['arxiv_id']}"
    if rec.get("inspire_id"):
        return f"recid:{rec['inspire_id']}"
    return f"title:{(rec.get('title') or '').strip().lower()}|year:{rec.get('year')}"

def _last_path_component(url: str) -> Optional[str]:
    try:
        path = urlparse(url).path
        if not path:
            return None
        return path.rstrip("/").split("/")[-1]
    except Exception:
        return None

async def _get_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # 简单退避：遇到 429 等待 5.2s 再试，最多 3 次
    for _ in range(3):
        r = await client.get(url, params=params)
        if r.status_code == 429:
            await asyncio.sleep(5.2)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}

async def _search_literature(client: httpx.AsyncClient, q: str, size: int, fields: str, sort: Optional[str] = None) -> List[Dict[str, Any]]:
    params = {"q": q, "size": max(1, min(size, 100)), "fields": fields}
    if sort:
        params["sort"] = sort
    data = await _get_json(client, f"{INSPIRE_BASE}/literature", params)
    hits = (data.get("hits") or {}).get("hits") or []
    out = []
    for h in hits:
        md = (h.get("metadata") or {})
        out.append(_extract_record_min(md))
    return out

async def _get_record_by_recid(client: httpx.AsyncClient, recid: str, fields: str) -> Optional[Dict[str, Any]]:
    try:
        data = await _get_json(client, f"{INSPIRE_BASE}/literature/{recid}", {"fields": fields})
        md = data.get("metadata") or data
        return _extract_record_min(md)
    except Exception:
        return None

# ------------------------------
# 召回查询（语义一致：不依赖精确短语）
# ------------------------------
def _fields_or_clause(tokens: List[str], include_fulltext: bool) -> str:
    toks = " OR ".join(tokens)
    subs = []
    if include_fulltext:
        subs.append(f"fulltext:({toks})")
    subs.append(f"abstracts.value:({toks})")
    subs.append(f"t:({toks})")
    subs.append(f"k:({toks})")
    return "(" + " OR ".join(subs) + ")"

def _fields_and_clause(tokens: List[str], include_fulltext: bool) -> str:
    toks = " AND ".join(tokens)
    subs = []
    if include_fulltext:
        subs.append(f"fulltext:({toks})")
    subs.append(f"abstracts.value:({toks})")
    subs.append(f"t:({toks})")
    subs.append(f"k:({toks})")
    return "(" + " OR ".join(subs) + ")"

def _semantic_query_variants(topic: str, include_fulltext: bool, start_year: int, end_year: int) -> Dict[str, str]:
    """
    返回四条语义召回查询：
      - and_terms: 概念组核心词 AND
      - syn_or   : 同义词/变体大 OR
      - title_syn: 题名增强 OR
      - kw_syn   : 关键词增强 OR
    """
    groups = _concept_groups_from_topic(topic)

    # OR 词表：所有组的所有变体
    or_tokens: List[str] = []
    for g in groups:
        or_tokens.extend(g)
    or_tokens = list(dict.fromkeys(or_tokens))[:60]

    # AND 词表：每组选若干“核心词”（避免选长短语）
    and_tokens: List[str] = []
    for g in groups:
        and_tokens.extend(_top_core_tokens(g, max_tokens=3))
    and_tokens = list(dict.fromkeys(and_tokens))[:12]
    if not and_tokens:
        and_tokens = or_tokens[:3] if or_tokens else []

    q_and = _fields_and_clause(and_tokens, include_fulltext)
    q_or  = _fields_or_clause(or_tokens, include_fulltext)
    q_title = f"t:({ ' OR '.join(or_tokens) })"
    q_kw    = f"k:({ ' OR '.join(or_tokens) })"

    date_clause = ""
    if start_year > 0 and end_year > 0 and end_year >= start_year:
        date_clause = f" and date:{start_year}->{end_year}"

    return {
        "and_terms": q_and + date_clause,
        "syn_or":    q_or  + date_clause,
        "title_syn": "(" + q_title + ")" + date_clause,
        "kw_syn":    "(" + q_kw    + ")" + date_clause,
    }

# ------------------------------
# RRF 融合
# ------------------------------
def _rrf_rank_dict(lst: List[Dict[str, Any]]) -> Dict[str, int]:
    return { _norm_key(r): i+1 for i, r in enumerate(lst) }

def _rrf_fuse_multi(rank_dicts: List[Dict[str, int]], weights: List[float], k: int = 60) -> List[Tuple[str, float]]:
    assert len(rank_dicts) == len(weights)
    keys: Set[str] = set()
    for rd in rank_dicts:
        keys |= set(rd.keys())
    out: List[Tuple[str, float]] = []
    for key in keys:
        s = 0.0
        for rd, w in zip(rank_dicts, weights):
            r = rd.get(key, math.inf)
            if math.isfinite(r):
                s += w / (k + r)
        out.append((key, s))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _rank_from_sorted_keys(keys_sorted_desc: List[str]) -> Dict[str, int]:
    return {k: i+1 for i, k in enumerate(keys_sorted_desc)}

# ------------------------------
# References 抓取与解析
# ------------------------------
def _parse_references(md: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    refs = md.get("references") or []
    cands: List[Dict[str, Optional[str]]] = []
    for r in refs:
        recid = None
        if isinstance(r.get("record"), dict) and r["record"].get("$ref"):
            recid = _last_path_component(r["record"]["$ref"])
        ref_meta = r.get("reference") or {}
        arx = ref_meta.get("arxiv_eprint")
        dois = ref_meta.get("dois") or []
        doi = None
        if isinstance(dois, list) and dois:
            val = dois[0].get("value")
            if val:
                doi = val
        title = ref_meta.get("title")
        cands.append({"recid": recid, "arxiv": arx, "doi": doi, "title": title})
    return cands

async def _fetch_references_for_recids(client: httpx.AsyncClient, recids: List[str], sem: asyncio.Semaphore) -> Dict[str, List[Dict[str, Optional[str]]]]:
    out: Dict[str, List[Dict[str, Optional[str]]]] = {}

    async def fetch_one(rid: str):
        async with sem:
            try:
                data = await _get_json(client, f"{INSPIRE_BASE}/literature/{rid}", {"fields": REF_ONLY_FIELDS})
                md = data.get("metadata") or data
                out[rid] = _parse_references(md)
            except Exception:
                out[rid] = []

    await asyncio.gather(*[fetch_one(r) for r in recids])
    return out

async def _resolve_ref_candidate(client: httpx.AsyncClient, cand: Dict[str, Optional[str]]) -> Optional[Dict[str, Any]]:
    if cand.get("recid"):
        return await _get_record_by_recid(client, cand["recid"], LIT_FIELDS_WITH_ABS)
    if cand.get("arxiv"):
        try:
            r = await client.get(f"{INSPIRE_BASE}/arxiv/{cand['arxiv']}")
            if r.status_code == 200:
                data = r.json()
                md = data.get("metadata") or data
                return _extract_record_min(md)
        except Exception:
            pass
    if cand.get("doi"):
        try:
            r = await client.get(f"{INSPIRE_BASE}/doi/{cand['doi']}")
            if r.status_code == 200:
                data = r.json()
                md = data.get("metadata") or data
                return _extract_record_min(md)
        except Exception:
            pass
    if cand.get("title"):
        try:
            # 不加引号，允许轻微标题差异
            params = {"q": f'title:{cand["title"]}', "size": 1, "fields": LIT_FIELDS_WITH_ABS}
            data = await _get_json(client, f"{INSPIRE_BASE}/literature", params)
            hits = (data.get("hits") or {}).get("hits") or []
            if hits:
                md = hits[0].get("metadata") or {}
                return _extract_record_min(md)
        except Exception:
            pass
    return None

# ------------------------------
# Tool 0: Author publications
# ------------------------------
@mcp.tool()
async def search_author_papers(author_name: str, size: int = 100, orcid: str = "") -> TextContent:
    """
    Search INSPIRE-HEP for an author and list their recent publications.
    Optional ORCID disambiguation. Returns authoritative fields only.
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
        try:
            if orcid:
                author_url = f"{INSPIRE_BASE}/orcid/{orcid}"
                r = await client.get(author_url)
                r.raise_for_status()
                a = r.json()
                recid = a.get("id")
                display_name = ((a.get("metadata") or {}).get("name") or {}).get("value", author_name)
            else:
                search_url = f"{INSPIRE_BASE}/authors"
                params = {"q": f'name.value:"{author_name}"', "size": 5, "fields": "control_number,name.value"}
                r = await client.get(search_url, params=params)
                r.raise_for_status()
                data = r.json()
                hits = (data.get("hits") or {}).get("hits") or []
                if not hits:
                    return TextContent(type="text", text=f"No author found with name '{author_name}' on INSPIRE-HEP.")
                hit = hits[0]
                recid = hit.get("id")
                display_name = ((hit.get("metadata") or {}).get("name") or {}).get("value", author_name)

            params = {
                "q": f"authors.recid:{recid}",
                "size": max(1, min(size, 200)),
                "sort": "mostrecent",
                "fields": LIT_FIELDS,
            }
            r = await client.get(f"{INSPIRE_BASE}/literature", params=params)
            r.raise_for_status()
            lit = r.json()
            pubs = (lit.get("hits") or {}).get("hits") or []
            if not pubs:
                return TextContent(type="text", text=f"Author '{display_name}' has no publications listed on INSPIRE-HEP.")

            lines = [f"Publications for author: {display_name}", "=" * 60]
            for i, h in enumerate(pubs, 1):
                md = h.get("metadata") or {}
                rec = _extract_record_min(md)
                authors_disp = "Unknown"
                if rec["authors"]:
                    authors_disp = ", ".join(rec["authors"][:3]) + (" et al." if len(rec["authors"]) > 3 else "")
                lines.append(
                    f"{i}. [{rec.get('year') or 'N/A'}] {rec.get('title')}\n"
                    f"   Authors : {authors_disp}\n"
                    f"   Citations: {rec.get('citation_count')}\n"
                    f"   arXiv   : {rec.get('arxiv_id') or 'N/A'} → {rec.get('arxiv_abs_url') or 'N/A'}\n"
                    f"   INSPIRE : {rec.get('inspire_url') or 'N/A'}"
                )
            return TextContent(type="text", text="\n".join(lines))

        except httpx.HTTPStatusError as e:
            return TextContent(type="text", text=f"Failed to fetch data: HTTP {e.response.status_code}")
        except httpx.ReadTimeout:
            return TextContent(type="text", text="Read timeout contacting INSPIRE-HEP.")
        except httpx.ConnectTimeout:
            return TextContent(type="text", text="Connect timeout contacting INSPIRE-HEP.")
        except Exception as e:
            return TextContent(type="text", text=f"An error occurred: {type(e).__name__}: {e}")

# ------------------------------
# Tool 1: Semantic topic search (+ reference expansion + classic fallback)
# ------------------------------
@mcp.tool()
async def search_topic_papers_semantic(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    include_fulltext: bool = False,
    as_json: bool = True,
    # 初级 RRF：四路语义召回 + 一路 mostcited
    w_and: float = 1.0,
    w_syn: float = 0.9,
    w_title: float = 0.8,
    w_kw: float = 0.5,
    w_cite: float = 1.0,
    rrf_k: int = 60,
    # 参考扩展参数
    expand_refs: bool = True,
    seeds_for_ref: int = 30,
    max_ref_per_seed: int = 25,
    max_ref_records: int = 200,
    # 最终 RRF（融合）
    w_seed: float = 1.0,
    w_cite_all: float = 0.7,
    w_text: float = 1.2,
    w_ref: float = 1.0,
    rrf_k_all: int = 60,
) -> TextContent:
    """
    语义主题检索（非精确短语）：多路召回（AND 核心词 / 同义词 OR / 题名 OR / 关键词 OR / mostcited），
    本地语义打分（标题+摘要），可选参考文献扩展，最终多信号 RRF 融合。
    返回仅权威字段 + 若干非权威调试字段（fused_score、text_score、ref_weight、via_seeds）。
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
        try:
            qs = _semantic_query_variants(topic, include_fulltext, start_year, end_year)
            size_each = max(1, min(size * 2, 100))

            # 四路语义召回 + 一路 mostcited
            rel_and, rel_syn, rel_title, rel_kw, cit_syn = await asyncio.gather(
                _search_literature(client, q=qs["and_terms"], size=size_each, fields=LIT_FIELDS_WITH_ABS, sort=None),
                _search_literature(client, q=qs["syn_or"],    size=size_each, fields=LIT_FIELDS_WITH_ABS, sort=None),
                _search_literature(client, q=qs["title_syn"], size=size_each, fields=LIT_FIELDS_WITH_ABS, sort=None),
                _search_literature(client, q=qs["kw_syn"],    size=size_each, fields=LIT_FIELDS_WITH_ABS, sort=None),
                _search_literature(client, q=qs["syn_or"],    size=size_each, fields=LIT_FIELDS_WITH_ABS, sort="mostcited"),
            )

            # 建立排名
            rank_and   = _rrf_rank_dict(rel_and)
            rank_syn   = _rrf_rank_dict(rel_syn)
            rank_title = _rrf_rank_dict(rel_title)
            rank_kw    = _rrf_rank_dict(rel_kw)
            rank_cit   = _rrf_rank_dict(cit_syn)

            # 种子融合
            fused_seed = _rrf_fuse_multi(
                [rank_and, rank_syn, rank_title, rank_kw, rank_cit],
                [w_and,    w_syn,    w_title,   w_kw,    w_cite],
                k=rrf_k
            )

            by_key_seed: Dict[str, Dict[str, Any]] = {}
            for r in rel_and + rel_syn + rel_title + rel_kw + cit_syn:
                by_key_seed.setdefault(_norm_key(r), r)

            seed_keys_ordered = [k for k, _ in fused_seed]
            seeds_main_keys = seed_keys_ordered[:max(size, 1)]
            seeds_for_ref_keys = seed_keys_ordered[:max(1, seeds_for_ref)]
            seed_base_score: Dict[str, float] = dict(fused_seed)

            # 初步结果（仅种子）
            results: Dict[str, Dict[str, Any]] = {}
            for kkey in seeds_main_keys:
                rec = dict(by_key_seed[kkey])
                rec["fused_score"] = round(seed_base_score.get(kkey, 0.0), 6)
                rec["text_score"] = round(_text_semantic_score(rec.get("title"), rec.get("abstract"), topic), 4)
                rec["ref_weight"] = 0.0
                rec["via_seeds"] = []
                results[kkey] = rec

            # 参考扩展（顺藤摸瓜）
            if expand_refs and seeds_for_ref_keys:
                seed_recids = []
                for key in seeds_for_ref_keys:
                    r = by_key_seed.get(key)
                    if r and r.get("inspire_id"):
                        seed_recids.append(str(r["inspire_id"]))

                sem = asyncio.Semaphore(8)
                refs_map = await _fetch_references_for_recids(client, seed_recids, sem)

                cand_list: List[Tuple[str, Dict[str, Optional[str]]]] = []
                for seed_rid, cands in refs_map.items():
                    if not cands:
                        continue
                    for c in cands[:max(1, max_ref_per_seed)]:
                        cand_list.append((seed_rid, c))

                cand_list = cand_list[:max(1, max_ref_records)]
                merged: Dict[str, Dict[str, Any]] = {}
                cand_sources: Dict[str, Set[str]] = {}

                async def resolve_one(seed_rid: str, c: Dict[str, Optional[str]]):
                    rec = await _resolve_ref_candidate(client, c)
                    if not rec:
                        return
                    key = _norm_key(rec)
                    if key not in merged:
                        merged[key] = rec
                    cand_sources.setdefault(key, set()).add(seed_rid)

                await asyncio.gather(*[resolve_one(seed, c) for seed, c in cand_list])

                # 语义打分 + 网络权重
                for key, rec in merged.items():
                    tscore = _text_semantic_score(rec.get("title"), rec.get("abstract"), topic)
                    if tscore <= 0.0:
                        continue
                    via = list(cand_sources.get(key, set()))
                    ref_w = 0.0
                    for seed_rid in via:
                        for kkey, seed_rec in by_key_seed.items():
                            if str(seed_rec.get("inspire_id")) == seed_rid:
                                ref_w += seed_base_score.get(kkey, 0.0)
                                break
                    if key in results:
                        results[key]["via_seeds"] = list(sorted(set(results[key].get("via_seeds", []) + via)))
                        results[key]["ref_weight"] = round(results[key].get("ref_weight", 0.0) + ref_w, 6)
                        results[key]["text_score"] = float(max(results[key].get("text_score", 0.0), tscore))
                    else:
                        newrec = dict(rec)
                        newrec["fused_score"] = 0.0
                        newrec["text_score"] = round(tscore, 4)
                        newrec["ref_weight"] = round(ref_w, 6)
                        newrec["via_seeds"] = via
                        results[key] = newrec

            # >>> Classic fallback: 用核心 AND 词 + mostcited “兜底”拉经典（确保 hep-th/0603001 被纳入）
            try:
                core_for_fallback: List[str] = []
                for g in _concept_groups_from_topic(topic):
                    core_for_fallback.extend(_top_core_tokens(g, max_tokens=3))
                core_for_fallback = list(dict.fromkeys([t for t in core_for_fallback if len(t) >= 3]))[:6]
                if core_for_fallback:
                    q_classic = _fields_and_clause(core_for_fallback, include_fulltext=False)
                    classic_list = await _search_literature(
                        client, q=q_classic, size=50, fields=LIT_FIELDS_WITH_ABS, sort="mostcited"
                    )
                    for rec in classic_list:
                        kkey = _norm_key(rec)
                        if kkey not in results:
                            results[kkey] = dict(rec)
                            results[kkey]["fused_score"] = 0.0
                            results[kkey]["text_score"]  = round(_text_semantic_score(rec.get("title"), rec.get("abstract"), topic), 4)
                            results[kkey]["ref_weight"]  = 0.0
                            results[kkey]["via_seeds"]   = []
            except Exception:
                pass

            # 最终排序：多路 RRF（种子、全体被引、语义得分、参考网络）
            keys_all = list(results.keys())
            rank_seed = {k: i+1 for i, k in enumerate(seeds_main_keys)}
            keys_cite = sorted(keys_all, key=lambda k: (results[k].get("citation_count") or -1), reverse=True)
            rank_cite_all = _rank_from_sorted_keys(keys_cite)
            keys_text = sorted(keys_all, key=lambda k: (results[k].get("text_score") or 0.0), reverse=True)
            rank_text = _rank_from_sorted_keys(keys_text)
            keys_ref = sorted(keys_all, key=lambda k: (results[k].get("ref_weight") or 0.0), reverse=True)
            rank_ref = _rank_from_sorted_keys(keys_ref)

            fused_all = _rrf_fuse_multi(
                [rank_seed, rank_cite_all, w_text and rank_text or {}, w_ref and rank_ref or {}],
                [w_seed,    w_cite_all,   w_text,                          w_ref],
                k=rrf_k_all
            )

            ordered = []
            for kkey, score in fused_all:
                if kkey in results:
                    results[kkey]["fused_score"] = round(score, 6)
                    ordered.append(results[kkey])

            final_list = ordered[:max(size, 1)]

            payload = {
                "query_used": qs,
                "sort_used": "semantic-hybrid(rrf) + classic-fallback + ref_expansion" if expand_refs else "semantic-hybrid(rrf) + classic-fallback",
                "weights": {
                    "seed_rrf": {"w_and": w_and, "w_syn": w_syn, "w_title": w_title, "w_kw": w_kw, "w_cite": w_cite, "rrf_k": rrf_k},
                    "final_rrf": {"w_seed": w_seed, "w_cite_all": w_cite_all, "w_text": w_text, "w_ref": w_ref, "rrf_k": rrf_k_all},
                },
                "ref_expansion": {
                    "enabled": bool(expand_refs),
                    "seeds_for_ref": seeds_for_ref,
                    "max_ref_per_seed": max_ref_per_seed,
                    "max_ref_records": max_ref_records,
                },
                "count": len(final_list),
                "results": final_list,
            }
            return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))

        except httpx.ReadTimeout:
            msg = {"error": "read_timeout"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except httpx.ConnectTimeout:
            msg = {"error": "connect_timeout"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except httpx.HTTPStatusError as e:
            msg = {"error": f"HTTP_{e.response.status_code}"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except Exception as e:
            msg = {"error": f"{type(e).__name__}: {e}"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))

# 便捷入口：默认启用参考扩展
@mcp.tool()
async def search_topic_papers_semantic_plus_refs(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    include_fulltext: bool = False,
    as_json: bool = True
) -> TextContent:
    return await search_topic_papers_semantic(
        topic=topic,
        size=size,
        start_year=start_year,
        end_year=end_year,
        include_fulltext=include_fulltext,
        as_json=as_json,
        expand_refs=True
    )

# ------------------------------
# Tool 2: BibTeX authenticity verification
# ------------------------------
_ARXIV_OLD = re.compile(r"[a-z\-]+\/\d{7}", re.IGNORECASE)   # e.g. hep-th/0405159
_ARXIV_NEW = re.compile(r"\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)
_DOI = re.compile(r"10\.\d{4,9}\/\S+", re.IGNORECASE)

def _parse_bibtex_entries(bibtex: str) -> List[Dict[str, str]]:
    entries = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bibtex, flags=re.DOTALL | re.IGNORECASE):
        body = m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.+?)[}\"]\s*,?", body, flags=re.DOTALL | re.IGNORECASE):
            key = fm.group(1).strip().lower()
            val = re.sub(r"\s+", " ", fm.group(2).strip())
            fields[key] = val
        entries.append({
            "key": m.group(2).strip(),
            "title": fields.get("title", ""),
            "author": fields.get("author", ""),
            "year": fields.get("year", ""),
            "doi": fields.get("doi", ""),
            "eprint": fields.get("eprint", "") or fields.get("arxiv", ""),
        })
    return entries

async def _resolve_one_entry(client: httpx.AsyncClient, ent: Dict[str, str]) -> Dict[str, Any]:
    title = ent.get("title", "")
    first_author = ent.get("author", "").split(" and ")[0] if ent.get("author") else ""
    year = ent.get("year", "")
    doi = ent.get("doi", "")
    eprint = ent.get("eprint", "")

    async def via_external_id() -> Optional[Dict[str, Any]]:
        if doi and _DOI.match(doi):
            r = await client.get(f"{INSPIRE_BASE}/doi/{doi}")
            if r.status_code == 200:
                return r.json()
        if eprint and (_ARXIV_NEW.match(eprint) or _ARXIV_OLD.match(eprint)):
            r = await client.get(f"{INSPIRE_BASE}/arxiv/{eprint}")
            if r.status_code == 200:
                return r.json()
        return None

    async def via_search() -> Optional[Dict[str, Any]]:
        terms = [f'title:"{title}"'] if title else []
        if first_author:
            terms.append(f'a:"{first_author}"')
        if year:
            terms.append(f"date:{year}")
        q = " and ".join(terms) if terms else ""
        if not q:
            return None
        params = {"q": q, "size": 1, "fields": LIT_FIELDS}
        r = await client.get(f"{INSPIRE_BASE}/literature", params=params)
        if r.status_code == 200:
            data = r.json()
            hits = (data.get("hits") or {}).get("hits") or []
            if hits:
                return hits[0]
        return None

    record = await via_external_id()
    if record is None:
        record = await via_search()

    report = {"bib_key": ent.get("key"), "found": False, "mismatch": [], "inspire_url": None, "arxiv_abs_url": None}
    if record:
        md = record.get("metadata") or record
        rec = _extract_record_min(md)
        report.update({
            "found": True,
            "inspire_url": rec.get("inspire_url"),
            "title_inspire": rec.get("title"),
            "year_inspire": rec.get("year"),
            "first_author_inspire": (rec.get("authors") or [""])[0] if rec.get("authors") else "",
            "arxiv_abs_url": rec.get("arxiv_abs_url"),
            "doi_equal": (doi.lower() in (md.get("dois", [{}])[0].get("value", "").lower() if md.get("dois") else "") ) if doi else None,
        })
        if ent.get("title") and rec.get("title") and ent["title"].strip().lower() != rec["title"].strip().lower():
            report["mismatch"].append("title")
        if year and rec.get("year") and str(year) != str(rec["year"]):
            report["mismatch"].append("year")
        if first_author and report.get("first_author_inspire") and first_author.strip().lower() not in report["first_author_inspire"].strip().lower():
            report["mismatch"].append("first_author")
    return report

@mcp.tool()
async def verify_bibtex_entries(bibtex: str, strict: bool = True, as_json: bool = True) -> TextContent:
    """
    Verify BibTeX entries against INSPIRE:
      - Prefer exact lookup via /api/doi/{doi} or /api/arxiv/{id};
      - Otherwise search by title+author(+year).
    """
    entries = _parse_bibtex_entries(bibtex)
    if not entries:
        return TextContent(type="text", text="No BibTeX entries detected.")

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
        try:
            reports = await asyncio.gather(*[_resolve_one_entry(client, e) for e in entries])
            overall_ok = all(r.get("found") and not r.get("mismatch") for r in reports) if strict else True
            payload = {"strict": strict, "overall_ok": overall_ok, "count": len(reports), "reports": reports}
            return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))
        except httpx.ReadTimeout:
            return TextContent(type="text", text="Read timeout contacting INSPIRE-HEP during BibTeX verification.")
        except httpx.ConnectTimeout:
            return TextContent(type="text", text="Connect timeout contacting INSPIRE-HEP during BibTeX verification.")
        except httpx.HTTPStatusError as e:
            return TextContent(type="text", text=f"HTTP error during BibTeX verification: {e.response.status_code}")
        except Exception as e:
            return TextContent(type="text", text=f"Error during BibTeX verification: {type(e).__name__}: {e}")

# ------------------------------
# 仅 mostcited（保留原功能）
# ------------------------------
@mcp.tool()
async def search_topic_papers_strict(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    include_fulltext: bool = False,
    as_json: bool = True,
    sort_by_citations: bool = True,
) -> TextContent:
    groups = _concept_groups_from_topic(topic)
    or_tokens: List[str] = []
    for g in groups:
        or_tokens.extend(g)
    or_tokens = list(dict.fromkeys(or_tokens))[:60]
    q = _fields_or_clause(or_tokens, include_fulltext)
    if start_year > 0 and end_year > 0 and end_year >= start_year:
        q += f" and date:{start_year}->{end_year}"

    params = {"q": q, "size": max(1, min(size, 50)), "fields": LIT_FIELDS}
    if sort_by_citations:
        params["sort"] = "mostcited"

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
        try:
            resp = await client.get(f"{INSPIRE_BASE}/literature", params=params)
            if resp.status_code == 429:
                payload = {"query_used": q, "sort_used": params.get("sort", "relevance"), "error": "rate_limited",
                           "message": "INSPIRE returned 429 (rate limited). Try later or narrow the query."}
                return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))
            resp.raise_for_status()
            data = resp.json()
        except httpx.ReadTimeout:
            msg = {"query_used": q, "sort_used": params.get("sort", "relevance"), "error": "read_timeout"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except httpx.ConnectTimeout:
            msg = {"query_used": q, "sort_used": params.get("sort", "relevance"), "error": "connect_timeout"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except httpx.HTTPStatusError as e:
            msg = {"query_used": q, "sort_used": params.get("sort", "relevance"), "error": f"HTTP_{e.response.status_code}"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except Exception as e:
            msg = {"query_used": q, "sort_used": params.get("sort", "relevance"), "error": f"{type(e).__name__}: {e}"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))

    hits = (data.get("hits") or {}).get("hits") or []
    results = []
    for h in hits:
        md = h.get("metadata") or {}
        results.append(_extract_record_min(md))

    if as_json:
        payload = {"query_used": q, "sort_used": params.get("sort", "relevance"), "count": len(results), "results": results}
        return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2))
    if not results:
        return TextContent(type="text", text=f"No results.\nQuery used: {q}\nSort: {params.get('sort', 'relevance')}")
    lines = [f"Query used: {q}", f"Sort: {params.get('sort', 'relevance')}", "=" * 72]
    for i, r in enumerate(results, 1):
        authors_disp = ", ".join(r.get("authors") or [])
        lines.append(
            f"{i}. [{r.get('earliest_date') or 'N/A'}] {r.get('title')}\n"
            f"   Citations     : {r.get('citation_count')}\n"
            f"   Authors       : {authors_disp}\n"
            f"   INSPIRE       : {r.get('inspire_url')}\n"
            f"   arXiv         : {r.get('arxiv_id')} ({r.get('arxiv_abs_url')})"
        )
    return TextContent(type="text", text="\n".join(lines))

# ------------------------------
# Main
# ------------------------------
def main():
    logger.info("Starting INSPIRE-HEP MCP server (semantic + core-AND + classic-fallback + reference expansion)")
    mcp.run("stdio")

if __name__ == "__main__":
    main()
