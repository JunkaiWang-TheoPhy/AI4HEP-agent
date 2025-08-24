# server.py — INSPIRE-HEP MCP server (hybrid ranking + bibtex verification + reference expansion)
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import logging
import httpx
import json
import re
import math
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Iterable, Set
from urllib.parse import urlparse

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------
# FastMCP
# ------------------------------
mcp = FastMCP()

# ------------------------------
# Shared HTTP client settings
# ------------------------------
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
DEFAULT_HEADERS = {"User-Agent": "inspire-mcp/3.0"}
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

# 扩展字段：供文本相关性打分（摘要）与参考解析
LIT_FIELDS_WITH_ABS = LIT_FIELDS + ",abstracts.value"
REF_ONLY_FIELDS = "references.record.$ref,references.reference.title,references.reference.arxiv_eprint,references.reference.dois.value"

# ------------------------------
# hep-th 常见英文同义词/变体（用于文本判分；可按需扩充）
# ------------------------------
SYNONYM_MAP = {
    "black hole": ["black hole", "black holes", "bh", "kerr", "schwarzschild"],
    "entropy": ["entropy", "entropic", "entropies", "bh entropy", "bekenstein", "hawking"],
    "black hole entropy": ["black hole entropy", "bekenstein-hawking entropy", "bh entropy", "gravitational entropy"],
    "quantum gravity": ["quantum gravity", "qg", "non-perturbative gravity", "quantum spacetime"],
    "string theory": ["string theory", "superstring", "m-theory", "heterotic", "type ii"],
    "holography": ["holography", "holographic", "ads/cft", "adscft", "gauge/gravity", "gauge gravity"],
    "conformal field theory": ["conformal field theory", "cft", "2d cft", "conformal symmetry"],
    "supersymmetry": ["supersymmetry", "susy", "n=4", "n=2", "superconformal"],
    "amplitude": ["scattering amplitude", "amplitude", "on-shell", "twistor", "bcfw"],
    "holographic entanglement entropy": [
        "holographic entanglement entropy", "entanglement entropy",
        "ryu takayanagi", "rt formula", "holographic entropy"
    ],
}

# ------------------------------
# Utilities
# ------------------------------
def _build_topic_query(
    topic: str,
    exact_phrase: bool,
    include_fulltext: bool,
    start_year: int,
    end_year: int,
) -> str:
    phrase = topic.strip().replace('"', '\\"')
    token = f'"{phrase}"' if exact_phrase else phrase
    subqs = [f"abstracts.value:{token}", f"t:{token}", f"k:{token}"]
    if include_fulltext:
        subqs.insert(0, f"fulltext:{token}")
    date_clause = ""
    if start_year > 0 and end_year > 0 and end_year >= start_year:
        # 'date' 是综合日期；如需更“早出现在记录”的语义，可换 'de:'（earliest_date）
        date_clause = f" and date:{start_year}->{end_year}"
    return "(" + " or ".join(subqs) + ")" + date_clause

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
        # 可能包含摘要（仅在请求 LIT_FIELDS_WITH_ABS 时）
        "abstract": ((md.get("abstracts") or [{}])[0].get("value") if md.get("abstracts") else None),
    }

def _norm_key(rec: Dict[str, Any]) -> str:
    # 用于去重：优先 arXiv，其次 inspire_id；兜底：标题+年份
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

def _simple_terms(topic: str) -> List[str]:
    t = topic.strip().lower()
    if t in SYNONYM_MAP:
        return list(dict.fromkeys([w.lower() for w in SYNONYM_MAP[t]]))
    # 否则做一些朴素扩展（复数/去空格）
    variants = [t]
    if " " in t:
        variants.append(t.replace(" ", ""))
    if not t.endswith("s") and len(t) > 3:
        variants.append(t + "s")
    return list(dict.fromkeys(variants))

def _text_relevance_score(title: Optional[str], abstract: Optional[str], topic: str) -> float:
    """
    0~1 之间：统计扩展词表在标题/摘要中的出现；标题加权更高
    """
    if not title and not abstract:
        return 0.0
    terms = _simple_terms(topic)
    if not terms:
        return 0.0

    def count_terms(text: str) -> int:
        low = text.lower()
        c = 0
        for w in terms:
            # 简单 contains，避免复杂分词；可按需改 regex 边界
            c += low.count(w)
        return c

    t_count = count_terms(title) if title else 0
    a_count = count_terms(abstract) if abstract else 0
    # 标题权重 2.0，摘要 1.0；并做 sigmoid-like 压缩避免极值
    raw = 2.0 * t_count + 1.0 * a_count
    if raw <= 0:
        return 0.0
    # 归一化：假定 8 命中 ≈ 接近 1
    return max(0.0, min(1.0, raw / 8.0))

def _rrf_fuse(rank_rel: Dict[str, int], rank_cit: Dict[str, int], w_rel=1.0, w_cite=1.0, k: int = 60) -> List[Tuple[str, float]]:
    keys = set(rank_rel) | set(rank_cit)
    scores = []
    for kkey in keys:
        r_rel = rank_rel.get(kkey, math.inf)
        r_cit = rank_cit.get(kkey, math.inf)
        s = 0.0
        if math.isfinite(r_rel):
            s += w_rel / (k + r_rel)
        if math.isfinite(r_cit):
            s += w_cite / (k + r_cit)
        scores.append((kkey, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _rrf_fuse_multi(rank_dicts: List[Dict[str, int]], weights: List[float], k: int = 60) -> List[Tuple[str, float]]:
    """通用 RRF 融合：支持多路排名"""
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

async def _get_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # 简单退避：尊重官方速率限制（每 IP 5s 窗口 15 次）——遇到429等待≥5s再试一次
    for attempt in range(3):
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

# 读取单条记录（可带 fields）
async def _get_record_by_recid(client: httpx.AsyncClient, recid: str, fields: str) -> Optional[Dict[str, Any]]:
    try:
        data = await _get_json(client, f"{INSPIRE_BASE}/literature/{recid}", {"fields": fields})
        md = data.get("metadata") or data
        return _extract_record_min(md)
    except Exception:
        return None

# 解析 references 字段，得到候选（recid / arxiv / doi / title）
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

# 并发抓取若干 recid 的 references 列表
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

# 将候选引用解析成最小权威记录（标题/作者/摘要/被引等）
async def _resolve_ref_candidate(client: httpx.AsyncClient, cand: Dict[str, Optional[str]]) -> Optional[Dict[str, Any]]:
    # 优先 recid → /literature/{id}
    if cand.get("recid"):
        return await _get_record_by_recid(client, cand["recid"], LIT_FIELDS_WITH_ABS)
    # 其次 arxiv → /api/arxiv/{eprint}
    if cand.get("arxiv"):
        try:
            r = await client.get(f"{INSPIRE_BASE}/arxiv/{cand['arxiv']}")
            if r.status_code == 200:
                data = r.json()
                md = data.get("metadata") or data
                return _extract_record_min(md)
        except Exception:
            pass
    # 再次 doi → /api/doi/{doi}
    if cand.get("doi"):
        try:
            r = await client.get(f"{INSPIRE_BASE}/doi/{cand['doi']}")
            if r.status_code == 200:
                data = r.json()
                md = data.get("metadata") or data
                return _extract_record_min(md)
        except Exception:
            pass
    # 兜底：title 精确短语检索
    if cand.get("title"):
        try:
            params = {"q": f'title:"{cand["title"].replace("\"","\\\"")}"', "size": 1, "fields": LIT_FIELDS_WITH_ABS}
            data = await _get_json(client, f"{INSPIRE_BASE}/literature", params)
            hits = (data.get("hits") or {}).get("hits") or []
            if hits:
                md = hits[0].get("metadata") or {}
                return _extract_record_min(md)
        except Exception:
            pass
    return None

def _rank_to_dict(keys_sorted_desc: List[str]) -> Dict[str, int]:
    """给定按“分数降序”的 key 列表，转为 RRF 所需的 rank 字典（1 开始）"""
    return {k: i + 1 for i, k in enumerate(keys_sorted_desc)}

# ------------------------------
# Tool 0: Author publications（微调保留）
# ------------------------------
@mcp.tool()
async def search_author_papers(author_name: str, size: int = 100, orcid: str = "") -> TextContent:
    """
    Search INSPIRE-HEP for an author and list their recent publications.
    Adds optional lookup by ORCID for disambiguation. Returns authoritative fields only.
    """
    timeout = DEFAULT_TIMEOUT
    async with httpx.AsyncClient(timeout=timeout, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
        try:
            # 优先 ORCID 精确匹配（若提供）
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

            # 拉文献
            fields = LIT_FIELDS
            params = {
                "q": f"authors.recid:{recid}",
                "size": max(1, min(size, 200)),
                "sort": "mostrecent",
                "fields": fields,
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
# Tool 1: Hybrid topic search (relevance + most cited via RRF)
#          + 可选参考文献扩展（顺藤摸瓜）
# ------------------------------
@mcp.tool()
async def search_topic_papers_hybrid(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    exact_phrase: bool = True,
    include_fulltext: bool = False,
    as_json: bool = True,
    w_rel: float = 1.0,
    w_cite: float = 1.0,
    rrf_k: int = 60,
    # ---- 新增参数（默认开启参考文献扩展；如要恢复老行为，传 expand_refs=False） ----
    expand_refs: bool = True,
    ref_depth: int = 1,                 # 仅实现 depth=1：从种子指向参考文献
    seeds_for_ref: int = 30,            # 参与扩展的前 N 篇种子
    max_ref_per_seed: int = 25,         # 每篇种子最多处理的参考条目
    max_ref_records: int = 200,         # 引入的参考候选总上限（解析后）
    topic_match_threshold: float = 0.15, # 文本相关性阈值（0~1）
    w_seed: float = 1.0,                # 融合权重：种子（原相关/被引）
    w_text: float = 0.8,                # 融合权重：文本相关性
    w_ref: float = 1.2,                 # 融合权重：参考网络权重（被多少高分种子引用）
    w_cite_all: float = 0.7,            # 融合权重：全体被引
    rrf_k_all: int = 60                 # 多路 RRF 的 k
) -> TextContent:
    """
    Hybrid mode: fetch relevance-sorted and mostcited-sorted lists from INSPIRE (authoritative),
    then locally fuse via weighted RRF. Optionally expand by traversing references of seeds,
    score by (text relevance + network hits + citations), and merge & deduplicate.
    Returns authoritative fields only; non-authoritative scores added for transparency.
    """
    q = _build_topic_query(topic, exact_phrase, include_fulltext, start_year, end_year)
    timeout = DEFAULT_TIMEOUT

    async with httpx.AsyncClient(timeout=timeout, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
        try:
            # 两路检索：相关性（默认）与最被引
            size_each = max(1, min(size * 2, 100))  # 多取一些，融合后截断
            rel_list, cit_list = await asyncio.gather(
                _search_literature(client, q=q, size=size_each, fields=LIT_FIELDS_WITH_ABS, sort=None),
                _search_literature(client, q=q, size=size_each, fields=LIT_FIELDS_WITH_ABS, sort="mostcited"),
            )

            # 建立排名
            rank_rel = { _norm_key(r): i+1 for i, r in enumerate(rel_list) }
            rank_cit = { _norm_key(r): i+1 for i, r in enumerate(cit_list) }

            # 先做基础融合，得到种子集合
            fused_seed = _rrf_fuse(rank_rel, rank_cit, w_rel=w_rel, w_cite=w_cite, k=rrf_k)
            by_key_seed: Dict[str, Dict[str, Any]] = {}
            for r in rel_list + cit_list:
                by_key_seed.setdefault(_norm_key(r), r)

            # 取前 size 作为主要种子清单，同时准备“所有种子+候选”集合
            seed_keys_ordered = [k for k, _ in fused_seed]
            seeds_main_keys = seed_keys_ordered[:max(size, 1)]
            seeds_for_ref_keys = seed_keys_ordered[:max(1, seeds_for_ref)]

            # 为每个种子记下基础 fused 分（用于参考网络加权）
            seed_base_score: Dict[str, float] = dict(fused_seed)

            # 组装初步结果（仅种子）
            results: Dict[str, Dict[str, Any]] = {}
            for kkey in seeds_main_keys:
                rec = dict(by_key_seed[kkey])
                rec["fused_score"] = round(seed_base_score.get(kkey, 0.0), 6)
                rec["text_score"] = round(_text_relevance_score(rec.get("title"), rec.get("abstract"), topic), 4)
                rec["ref_weight"] = 0.0
                rec["via_seeds"] = []
                results[kkey] = rec

            # ---- 参考文献扩展（depth=1）----
            if expand_refs and ref_depth >= 1 and seeds_for_ref_keys:
                # 取 seeds_for_ref_keys 对应 inspire_id，用于抓 references
                seed_recids = []
                for key in seeds_for_ref_keys:
                    r = by_key_seed.get(key)
                    if r and r.get("inspire_id"):
                        seed_recids.append(str(r["inspire_id"]))

                # 并发抓 references
                sem = asyncio.Semaphore(8)  # 控制抓取并发，避免触发严格限速
                refs_map = await _fetch_references_for_recids(client, seed_recids, sem)

                # 整理候选（限制每篇最多 max_ref_per_seed，整体最多 max_ref_records）
                cand_list: List[Tuple[str, Dict[str, Optional[str]]]] = []  # (seed_recid, cand)
                for seed_rid, cands in refs_map.items():
                    if not cands:
                        continue
                    take = cands[:max(1, max_ref_per_seed)]
                    for c in take:
                        cand_list.append((seed_rid, c))

                # 解析候选 → 权威记录（摘要等），并统计“来自哪些种子”
                # 去重键：优先 recid / arxiv / doi / title
                def cand_key(c: Dict[str, Optional[str]]) -> str:
                    if c.get("recid"):
                        return f"recid:{c['recid']}"
                    if c.get("arxiv"):
                        return f"arxiv:{c['arxiv']}"
                    if c.get("doi"):
                        return f"doi:{c['doi'].lower()}"
                    if c.get("title"):
                        return f"title:{c['title'].strip().lower()}"
                    return json.dumps(c, sort_keys=True)

                merged: Dict[str, Dict[str, Any]] = {}
                cand_sources: Dict[str, Set[str]] = {}

                async def resolve_one(seed_rid: str, c: Dict[str, Optional[str]]):
                    rec = await _resolve_ref_candidate(client, c)
                    if not rec:
                        return
                    # 舍弃自身是种子的（会合并，但保留网络来源供加权）
                    key = _norm_key(rec)
                    if key not in merged:
                        merged[key] = rec
                    cand_sources.setdefault(key, set()).add(seed_rid)

                # 控制解析上限
                capped = cand_list[:max(1, max_ref_records)]
                await asyncio.gather(*[resolve_one(seed, c) for seed, c in capped])

                # 对解析出的候选做文本相关性打分 + 过滤阈值
                ref_items_scored: Dict[str, Dict[str, Any]] = {}
                for key, rec in merged.items():
                    tscore = _text_relevance_score(rec.get("title"), rec.get("abstract"), topic)
                    if tscore >= max(0.0, min(1.0, topic_match_threshold)):
                        newrec = dict(rec)
                        newrec["text_score"] = round(tscore, 4)
                        # ref_weight：由多少高分种子指向来衡量；使用种子基础 fused 分累加
                        via = list(cand_sources.get(key, set()))
                        ref_w = 0.0
                        for seed_rid in via:
                            # 找到该 seed_rid 对应的 key，以便取 fused 分
                            # 构建 recid->key 映射
                            # 简单做法：在 by_key_seed 里扫描一次（种子数量有限）
                            for kkey, seed_rec in by_key_seed.items():
                                if str(seed_rec.get("inspire_id")) == seed_rid:
                                    ref_w += seed_base_score.get(kkey, 0.0)
                                    break
                        newrec["ref_weight"] = round(ref_w, 6)
                        newrec["via_seeds"] = via
                        ref_items_scored[key] = newrec

                # 合并“参考候选”进总集合（若已存在为种子，则仅补充网络/文本信息作加权）
                for key, rec in ref_items_scored.items():
                    if key in results:
                        # 种子上追加来源与网络权重
                        results[key]["via_seeds"] = list(sorted(set(results[key].get("via_seeds", []) + rec.get("via_seeds", []))))
                        results[key]["ref_weight"] = round(results[key].get("ref_weight", 0.0) + rec.get("ref_weight", 0.0), 6)
                        # 文本分取较高者
                        results[key]["text_score"] = float(max(results[key].get("text_score", 0.0), rec.get("text_score", 0.0)))
                    else:
                        # 参考候选（非种子）先放入，稍后做最终融合排序
                        # 初始化 fused_score=0（稍后统一用多路 RRF 产生）
                        rec["fused_score"] = 0.0
                        results[key] = rec

            # ---- 最终排序：多路 RRF（种子位置、全体被引、文本相关性、参考网络权重）----
            # 1) rank_seed：仅种子有名次（越靠前越好）
            rank_seed = {k: i+1 for i, k in enumerate(seeds_main_keys)}
            # 2) rank_cite_all：全体按 citation_count 降序
            keys_all = list(results.keys())
            keys_cite = sorted(keys_all, key=lambda k: (results[k].get("citation_count") or -1), reverse=True)
            rank_cite_all = _rank_to_dict(keys_cite)
            # 3) rank_text：按 text_score 降序
            keys_text = sorted(keys_all, key=lambda k: (results[k].get("text_score") or 0.0), reverse=True)
            rank_text = _rank_to_dict(keys_text)
            # 4) rank_ref：按 ref_weight 降序
            keys_ref = sorted(keys_all, key=lambda k: (results[k].get("ref_weight") or 0.0), reverse=True)
            rank_ref = _rank_to_dict(keys_ref)

            fused_all = _rrf_fuse_multi(
                [rank_seed, rank_cite_all, rank_text, rank_ref],
                [w_seed,    w_cite_all,   w_text,    w_ref],
                k=rrf_k_all
            )

            # 回填最终 fused_score，并输出前 size
            ordered = []
            for kkey, score in fused_all:
                if kkey in results:
                    results[kkey]["fused_score"] = round(score, 6)
                    ordered.append(results[kkey])

            # 截断
            final_list = ordered[:max(size, 1)]

            payload = {
                "query_used": q,
                "sort_used": "hybrid(rrf) + ref_expansion" if expand_refs else "hybrid(rrf)",
                "weights": {
                    "seed_rrf": {"w_rel": w_rel, "w_cite": w_cite, "rrf_k": rrf_k},
                    "final_rrf": {"w_seed": w_seed, "w_cite_all": w_cite_all, "w_text": w_text, "w_ref": w_ref, "rrf_k": rrf_k_all},
                },
                "ref_expansion": {
                    "enabled": bool(expand_refs),
                    "depth": ref_depth,
                    "seeds_for_ref": seeds_for_ref,
                    "max_ref_per_seed": max_ref_per_seed,
                    "max_ref_records": max_ref_records,
                    "topic_match_threshold": topic_match_threshold,
                },
                "count": len(final_list),
                "results": final_list,
            }
            return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))

        except httpx.ReadTimeout:
            msg = {"query_used": q, "sort_used": "hybrid(rrf)", "error": "read_timeout"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except httpx.ConnectTimeout:
            msg = {"query_used": q, "sort_used": "hybrid(rrf)", "error": "connect_timeout"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except httpx.HTTPStatusError as e:
            msg = {"query_used": q, "sort_used": "hybrid(rrf)", "error": f"HTTP_{e.response.status_code}"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))
        except Exception as e:
            msg = {"query_used": q, "sort_used": "hybrid(rrf)", "error": f"{type(e).__name__}: {e}"}
            return TextContent(type="text", text=json.dumps(msg, ensure_ascii=False, indent=2) if as_json else str(msg))

# ---- 便捷入口：默认开启参考文献扩展 ----
@mcp.tool()
async def search_topic_papers_hybrid_plus_refs(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    exact_phrase: bool = True,
    include_fulltext: bool = False,
    as_json: bool = True
) -> TextContent:
    return await search_topic_papers_hybrid(
        topic=topic,
        size=size,
        start_year=start_year,
        end_year=end_year,
        exact_phrase=exact_phrase,
        include_fulltext=include_fulltext,
        as_json=as_json,
        expand_refs=True
    )

# ------------------------------
# Tool 2: BibTeX authenticity verification（保持原有、微调健壮性）
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
    Returns a per-entry report with mismatches.
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
# Tool 3 (新增): 修复 .tex 中误写的 \\section / \\begin / \\textbf 等命令
#   - 仅当 \\ 后紧跟字母或 @ 时收缩为单反斜杠；
#   - 跳过 verbatim / lstlisting 环境与 \verb... 内联内容；
#   - 不影响真正的换行命令 \\。
# ------------------------------
# ------------------------------
# Tool 4: 编译 LaTeX 到 PDF（支持传文件/传源码），固定输出在 <server_dir>/latex_gen
#   - 优先 latexmk；无则尝试 tectonic；再退化为 (xe|pdf|lua)latex 多次编译 + (biber|bibtex)
#   - 默认为非交互/不中断模式；返回 pdf 路径与日志尾部，便于排错
# ------------------------------
@mcp.tool()
def compile_tex_to_pdf(
    tex_filename: str = "",
    tex_source: str = "",           # 若传源码，此项非空并要求设置 jobname
    jobname: str = "",
    base_dir: str = "",             # 可显式指定根目录；默认 = server.py 同级
    out_dir: str = "",              # 可显式指定输出目录；默认 = <base_dir>/latex_gen
    engine: str = "auto",           # auto|latexmk|tectonic|direct
    pdf_engine: str = "xelatex",    # direct 时使用：xelatex|pdflatex|lualatex
    bib: str = "auto",              # auto|biber|bibtex|none
    runs: int = 2,                  # direct 时编译轮数
    shell_escape: bool = False,     # direct 时支持；latexmk/tectonic 下默认关闭以保证安全
    texinputs_extra: str = "",      # 额外 TEXINPUTS 路径，多个用操作系统分隔符连接
    timeout_sec: int = 300,
    as_json: bool = True
) -> TextContent:
    import subprocess, shutil, time, tempfile

    def _resolve_base_dir() -> Path:
        if base_dir:
            return Path(base_dir).resolve()
        env = os.environ.get("MCP_LATEX_OUTPUT_DIR")
        if env:
            return Path(env).resolve()
        return Path(__file__).parent.resolve()

    def _resolve_out_dir(_base: Path) -> Path:
        p = Path(out_dir).resolve() if out_dir else (_base / "latex_gen")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _which(cmd: str) -> Optional[str]:
        return shutil.which(cmd)

    def _run(cmd: List[str], cwd: Path, env: Dict[str, str]) -> Tuple[int, str]:
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=max(30, timeout_sec), check=False, text=True, encoding="utf-8", errors="ignore"
            )
            return proc.returncode, proc.stdout
        except subprocess.TimeoutExpired as e:
            return 124, (e.stdout or "") + "\n[compile] TIMEOUT"

    base = _resolve_base_dir()
    outp = _resolve_out_dir(base)

    # 1) 确定 tex 源
    tex_path: Optional[Path] = None
    if tex_source and not tex_filename:
        if not jobname:
            return TextContent(type="text", text=json.dumps({"ok": False, "error": "jobname required when using tex_source"}, ensure_ascii=False))
        tex_path = (outp / f"{jobname}.tex").resolve()
        tex_path.write_text(tex_source, encoding="utf-8")
    else:
        if not tex_filename:
            return TextContent(type="text", text=json.dumps({"ok": False, "error": "tex_filename or tex_source must be provided"}, ensure_ascii=False))
        p = Path(tex_filename)
        if not p.is_absolute():
            p = base / p
        tex_path = p.resolve()
        if not tex_path.exists():
            return TextContent(type="text", text=json.dumps({"ok": False, "error": f"tex file not found: {str(tex_path)}"}, ensure_ascii=False))
        if not jobname:
            jobname = tex_path.stem

    workdir = tex_path.parent
    pdf_path = (outp / f"{jobname}.pdf").resolve()

    # 2) 环境变量（TEXINPUTS 追加，并保留系统默认路径；注意加结尾分隔符表示“再搜索默认”）
    env = os.environ.copy()
    extras = [s for s in (texinputs_extra or "").split(os.pathsep) if s.strip()]
    add_paths = [str(workdir), str(outp)] + extras
    # kpathsea 规则：末尾加路径分隔符表示“包含系统默认查找路径”
    env["TEXINPUTS"] = (os.pathsep.join(add_paths) + os.pathsep +
                        env.get("TEXINPUTS", ""))

    # 3) 编译策略
    used = []
    start = time.time()
    log_all = ""

    # 3.1 latexmk（优先；更稳）—— 不强行注入 --shell-escape，避免安全问题
    if engine in ("auto", "latexmk") and _which("latexmk"):
        used.append("latexmk")
        # latexmk 引擎选择
        eng_flag = {"xelatex": "-xelatex", "lualatex": "-lualatex", "pdflatex": "-pdf"}.get(pdf_engine, "-xelatex")
        cmd = ["latexmk", eng_flag, f"-outdir={str(outp)}"]
        if jobname:
            # 某些 latexmk 旧版本对 -jobname 支持不佳；兼容性起见：仅在存在时传
            cmd += [f"-jobname={jobname}"]
        if bib == "biber":
            cmd += ["-use-biber"]
        elif bib == "bibtex":
            cmd += ["-bibtex"]
        # 安静一些但保留错误
        cmd += ["-file-line-error", "-interaction=nonstopmode", tex_path.name]
        code, out = _run(cmd, cwd=workdir, env=env)
        log_all += f"\n[latexmk cmd] {' '.join(cmd)}\n" + out
        if code == 0 and pdf_path.exists():
            elapsed = round(time.time() - start, 3)
            tail = "\n".join(out.splitlines()[-80:])
            payload = {
                "ok": True, "engine_used": "latexmk",
                "pdf": str(pdf_path), "out_dir": str(outp), "work_dir": str(workdir),
                "jobname": jobname, "elapsed_sec": elapsed, "log_tail": tail
            }
            return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))
        # 否则继续尝试下一策略

    # 3.2 tectonic（自带依赖管理；不支持 shell-escape）
    if engine in ("auto", "tectonic") and not shell_escape and _which("tectonic"):
        used.append("tectonic")
        cmd = ["tectonic", str(tex_path), "-o", str(outp)]
        code, out = _run(cmd, cwd=workdir, env=env)
        log_all += f"\n[tectonic cmd] {' '.join(cmd)}\n" + out
        if code == 0 and pdf_path.exists():
            elapsed = round(time.time() - start, 3)
            tail = "\n".join(out.splitlines()[-80:])
            payload = {
                "ok": True, "engine_used": "tectonic",
                "pdf": str(pdf_path), "out_dir": str(outp), "work_dir": str(workdir),
                "jobname": jobname, "elapsed_sec": elapsed, "log_tail": tail
            }
            return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))

    # 3.3 直接调 (xe|pdf|lua)latex (+ biber/bibtex)，可选 --shell-escape
    used.append("direct")
    engine_bin = {"xelatex": "xelatex", "pdflatex": "pdflatex", "lualatex": "lualatex"}.get(pdf_engine, "xelatex")
    if not _which(engine_bin):
        return TextContent(type="text", text=json.dumps({"ok": False, "error": f"{engine_bin} not found in PATH", "tried": used}, ensure_ascii=False))

    # 先跑一次，必要时插入 biber/bibtex，再补跑若干次
    base_cmd = [engine_bin, "-interaction=nonstopmode", "-file-line-error"]
    if shell_escape:
        base_cmd += ["-shell-escape"]
    base_cmd += ["-synctex=1", "-halt-on-error", f"-output-directory={str(outp)}"]
    # Windows 下 -jobname 支持可靠；统一传
    if jobname:
        base_cmd += [f"-jobname={jobname}"]
    code1, out1 = _run(base_cmd + [tex_path.name], cwd=workdir, env=env)
    log_all += f"\n[{engine_bin} #1] {' '.join(base_cmd)} {tex_path.name}\n" + out1

    # 自动选择 biber / bibtex
    if bib != "none":
        use_biber = (bib == "biber")
        use_bibtex = (bib == "bibtex")
        if bib == "auto":
            # 若 outdir 下生成了 .bcf 则优先 biber；否则若有 .aux 用 bibtex
            bcf = (outp / f"{jobname}.bcf")
            aux = (outp / f"{jobname}.aux")
            if bcf.exists() and _which("biber"):
                use_biber = True
            elif aux.exists() and _which("bibtex"):
                use_bibtex = True

        if use_biber and _which("biber"):
            code_b, out_b = _run(["biber", jobname], cwd=outp, env=env)  # biber 在 outdir 下执行更稳
            log_all += f"\n[biber] biber {jobname}\n" + out_b
        elif use_bibtex and _which("bibtex"):
            # bibtex 需要在 outdir/ 下运行
            code_b, out_b = _run(["bibtex", jobname], cwd=outp, env=env)
            log_all += f"\n[bibtex] bibtex {jobname}\n" + out_b
        # 无 biber/bibtex 可用则忽略

    # 追加编译轮次
    remaining = max(1, int(runs))
    last_code, last_out = code1, out1
    for i in range(2, remaining + 2):
        code_i, out_i = _run(base_cmd + [tex_path.name], cwd=workdir, env=env)
        log_all += f"\n[{engine_bin} #{i}] {' '.join(base_cmd)} {tex_path.name}\n" + out_i
        last_code, last_out = code_i, out_i

    elapsed = round(time.time() - start, 3)
    ok = pdf_path.exists()
    tail = "\n".join((last_out or log_all).splitlines()[-120:])
    payload = {
        "ok": bool(ok), "engine_used": f"direct:{engine_bin}",
        "pdf": str(pdf_path) if ok else None,
        "out_dir": str(outp), "work_dir": str(workdir),
        "jobname": jobname, "elapsed_sec": elapsed,
        "tried": used, "log_tail": tail
    }
    # 若失败，附上一点提示
    if not ok:
        payload["hint"] = "Check missing packages/fonts; see log_tail for the first fatal error."
    return TextContent(type="text", text=json.dumps(payload, ensure_ascii=False, indent=2) if as_json else str(payload))


# ------------------------------
# Tool 5: 先修复误写命令再编译（便捷一键）
# ------------------------------
@mcp.tool()
def fix_then_compile_tex(
    tex_filename: str,
    base_dir: str = "",
    out_dir: str = "",
    engine: str = "auto",
    pdf_engine: str = "xelatex",
    bib: str = "auto",
    runs: int = 2,
    shell_escape: bool = False,
    texinputs_extra: str = "",
    timeout_sec: int = 300,
    as_json: bool = True
) -> TextContent:
    # 先干跑一遍修复（非 dry_run）
    _ = fix_tex_commands_in_file(tex_filename=tex_filename, base_dir=base_dir, dry_run=False)
    # 再调用编译
    return compile_tex_to_pdf(
        tex_filename=tex_filename, tex_source="", jobname="",
        base_dir=base_dir, out_dir=out_dir,
        engine=engine, pdf_engine=pdf_engine, bib=bib, runs=runs,
        shell_escape=shell_escape, texinputs_extra=texinputs_extra,
        timeout_sec=timeout_sec, as_json=as_json
    )

@mcp.tool()
def fix_tex_commands_in_file(
    tex_filename: str,
    base_dir: Optional[str] = None,
    dry_run: bool = False
) -> TextContent:
    """
    将 .tex 文件中误写成双反斜杠的命令（如 \\section、\\begin、\\textbf、\\cite 等）自动修复为单反斜杠。
    仅当“\\ 后紧跟字母或 @”时才收缩（避免误改换行命令“\\”），并且跳过 verbatim / lstlisting 环境与 \\verb 内联。

    - tex_filename: 文件名或绝对路径；若为相对路径则基于 base_dir 或 MCP_LATEX_OUTPUT_DIR。
    - base_dir: 可显式指定根目录；优先于 MCP_LATEX_OUTPUT_DIR。
    - dry_run: True 时仅返回将要修改的次数与示例，不写回文件。
    """
    # 计算目标路径
    out_dir = Path(base_dir or os.environ.get("MCP_LATEX_OUTPUT_DIR", "."))
    p = Path(tex_filename)
    if not p.is_absolute():
        p = out_dir / p
    p = p.resolve()

    if not p.exists():
        return TextContent(type="text", text=json.dumps({"ok": False, "error": f"tex file not found: {str(p)}"}, ensure_ascii=False))

    text = p.read_text(encoding="utf-8", errors="ignore")

    # —— 掩蔽 verbatim / lstlisting 环境与 \verb 内联，避免误改 —— #
    masks: List[str] = []
    token_prefix = "@@MCPMASK"
    masked = text

    def _mask_env(name: str, s: str) -> str:
        # 非贪婪匹配 \begin{name} ... \end{name}
        pat = re.compile(rf'\\begin\{{{name}\}}.*?\\end\{{{name}\}}', re.DOTALL | re.IGNORECASE)
        def repl(m):
            token = f"{token_prefix}{len(masks)}@@"
            masks.append(m.group(0))
            return token
        return pat.sub(repl, s)

    for env in ("verbatim", "lstlisting"):
        masked = _mask_env(env, masked)

    # \verb<delim>...<delim> 形式（delim 为任意非空白单字符）
    pat_verb = re.compile(r'\\verb(?P<d>[^A-Za-z0-9\s])(?:(?!\n).)*?(?P=d)', re.DOTALL)
    def repl_verb(m):
        token = f"{token_prefix}{len(masks)}@@"
        masks.append(m.group(0))
        return token
    masked = pat_verb.sub(repl_verb, masked)

    # —— 真正的修复：仅当 \\ 后紧跟字母或 @ 才收缩为单反斜杠 —— #
    fixed, n = re.subn(r'\\\\(?=[A-Za-z@])', r'\\', masked)

    # 还原掩蔽块
    for i, seg in enumerate(masks):
        fixed = fixed.replace(f"{token_prefix}{i}@@", seg)

    result = {
        "ok": True,
        "file": str(p),
        "fixed_occurrences": n,
        "dry_run": bool(dry_run),
    }

    if not dry_run and n > 0:
        p.write_text(fixed, encoding="utf-8")

    return TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))

# ------------------------------
# 保留：strict 版本（仅 mostcited 服务器端排序）
# ------------------------------
@mcp.tool()
async def search_topic_papers_strict(
    topic: str,
    size: int = 20,
    start_year: int = 0,
    end_year: int = 0,
    exact_phrase: bool = True,
    include_fulltext: bool = False,
    as_json: bool = True,
    sort_by_citations: bool = True,
) -> TextContent:
    q = _build_topic_query(topic, exact_phrase, include_fulltext, start_year, end_year)
    timeout = DEFAULT_TIMEOUT
    params = {"q": q, "size": max(1, min(size, 50)), "fields": LIT_FIELDS}
    if sort_by_citations:
        params["sort"] = "mostcited"

    async with httpx.AsyncClient(timeout=timeout, headers=DEFAULT_HEADERS, limits=HTTP_LIMITS) as client:
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
    logger.info("Starting INSPIRE-HEP MCP server (with reference expansion)")
    mcp.run("stdio")

if __name__ == "__main__":
    main()
