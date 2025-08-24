from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import logging
import httpx

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 MCP 服务器实例
mcp = FastMCP()

# -----------------------------
# 工具1：搜索作者的论文列表
# -----------------------------
@mcp.tool()
async def search_author_papers(author_name: str) -> TextContent:
    """
    Search for an author on INSPIRE-HEP and return their publication list.

    Args:
        author_name: The full name or last name of the author (e.g., "Einstein", "Albert Einstein")

    Returns:
        A formatted text list of the author's papers.
    """
    logger.info(f"Searching for author: {author_name}")

    async with httpx.AsyncClient() as client:
        try:
            search_url = "https://inspirehep.net/api/authors"
            params = {"q": author_name, "size": 5}
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data["hits"]["hits"]:
                return TextContent(type="text", text=f"No author found with name '{author_name}' on INSPIRE-HEP.")

            hit = data["hits"]["hits"][0]
            recid = hit["id"]
            display_name = hit["metadata"].get("name", {}).get("value", "Unknown")
            logger.info(f"Found author: {display_name} (recid: {recid})")

            literature_url = "https://inspirehep.net/api/literature"
            literature_params = {
                "q": f"authors.recid:{recid}",
                "size": 100,
                "sort": "mostrecent"
            }
            response = await client.get(literature_url, params=literature_params)
            response.raise_for_status()
            papers_data = response.json()

            if not papers_data["hits"]["hits"]:
                return TextContent(type="text", text=f"Author '{display_name}' has no publications listed.")

            result_lines = [f"Publications for author: {display_name}\n" + "=" * 60]
            for i, paper_hit in enumerate(papers_data["hits"]["hits"], 1):
                metadata = paper_hit["metadata"]
                title = metadata["titles"][0]["title"]

                year = "N/A"
                if "publication_info" in metadata and len(metadata["publication_info"]) > 0:
                    year = metadata["publication_info"][0].get("year", "N/A")
                elif "preprint_date" in metadata:
                    year = metadata["preprint_date"][:4]
                elif "earliest_date" in metadata:
                    year = metadata["earliest_date"][:4]

                citation_count = metadata.get("citation_count", 0)

                arxiv_id = "N/A"
                arxiv_link = "N/A"
                if metadata.get("arxiv_eprints"):
                    arxiv_id = metadata["arxiv_eprints"][0]["value"]
                    arxiv_link = f"https://arxiv.org/abs/{arxiv_id}"

                result_lines.append(
                    f"{i}. [{year}] {title} (Citations: {citation_count})\n   arXiv: {arxiv_id} → {arxiv_link}"
                )

            return TextContent(type="text", text="\n".join(result_lines))

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return TextContent(type="text", text=f"Failed to fetch data: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            return TextContent(type="text", text=f"An error occurred: {type(e).__name__}: {e}")


# -----------------------------
# 工具2：根据 arXiv ID 获取论文摘要
# -----------------------------
@mcp.tool()
async def get_paper_abstract(arxiv_id: str) -> TextContent:
    """
    Fetch the title and abstract of a paper from arXiv using its identifier.

    Args:
        arxiv_id: The arXiv identifier, e.g. "1507.00123", "0704.0001", or "hep-th/9901001"

    Returns:
        Formatted text containing the paper's title and abstract.
    """
    logger.info(f"Fetching paper abstract for arXiv ID: {arxiv_id}")

    async with httpx.AsyncClient() as client:
        try:
            # 使用 INSPIRE-HEP API 查询 arXiv 论文
            url = "https://inspirehep.net/api/literature"
            params = {"q": f"arxiv_eprints.value:{arxiv_id}", "size": 1}
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data["hits"]["hits"]:
                return TextContent(type="text", text=f"No paper found with arXiv ID '{arxiv_id}'.")

            hit = data["hits"]["hits"][0]
            metadata = hit["metadata"]

            title = metadata["titles"][0]["title"]
            abstract = metadata.get("abstracts", [{}])[0].get("value", "No abstract available.")
            # 清理换行符，确保输出整洁
            abstract = abstract.replace("\n", " ").strip()

            # 构造 arXiv 链接
            arxiv_link = f"https://arxiv.org/abs/{arxiv_id}"

            result = (
                f"Title: {title}\n"
                f"arXiv: {arxiv_id} → {arxiv_link}\n"
                f"Abstract: {abstract}"
            )
            return TextContent(type="text", text=result)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return TextContent(type="text", text=f"Failed to fetch paper: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}: {e}")
            return TextContent(type="text", text=f"An error occurred: {type(e).__name__}: {e}")



# ==================
from typing import Optional, List, Dict
import asyncio
import numpy as np

# 全局缓存 embedding 模型，避免重复加载
_EMB_MODEL = None
_EMB_MODEL_NAME = None

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def _get_embedder(model_name: str):
    """惰性加载 sentence-transformers 模型；model_name 可切换。"""
    global _EMB_MODEL, _EMB_MODEL_NAME
    if _EMB_MODEL is not None and _EMB_MODEL_NAME == model_name:
        return _EMB_MODEL
    from sentence_transformers import SentenceTransformer
    _EMB_MODEL = SentenceTransformer(model_name)
    _EMB_MODEL_NAME = model_name
    return _EMB_MODEL

async def _inspr_fetch_candidates(
    client: httpx.AsyncClient,
    topic: str,
    fetch_k: int,
    start_year: Optional[int],
    end_year: Optional[int],
    include_fulltext: bool,
) -> List[Dict]:
    """用多条‘宽松查询’从 INSPIRE 拉候选，去重后最多返回 fetch_k 条。"""
    phrase = topic.strip().replace('"', '\\"')
    date_clause = ""
    if start_year is not None and end_year is not None:
        date_clause = f" and date:{start_year}->{end_year}"

    # 由强到弱的多级召回（避免 0 命中）
    q_list = [
        f'(abstracts.value:"{phrase}" or t:"{phrase}" or k:"{phrase}"){date_clause}',
        f'(abstracts.value:{phrase} or t:{phrase} or k:{phrase}){date_clause}',
        f'"{phrase}"{date_clause}',
    ]
    if include_fulltext:
        q_list.insert(0, f'fulltext:"{phrase}"{date_clause}')

    fields = ",".join([
        "titles.title",
        "authors.full_name",
        "arxiv_eprints.value",
        "earliest_date",
        "citation_count",
        "abstracts.value"
    ])

    seen = set()
    out: List[Dict] = []
    literature_url = "https://inspirehep.net/api/literature"

    for q in q_list:
        if len(out) >= fetch_k:
            break
        params = {
            "q": q,
            "size": min(1000, fetch_k),  # 单页上限 1000
            "fields": fields,
        }
        try:
            resp = await client.get(literature_url, params=params)
            if resp.status_code == 429:
                logger.warning("Rate limited (429). Backing off 5s and retrying once.")
                await asyncio.sleep(5)
                resp = await client.get(literature_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            for h in hits:
                hid = h.get("id")
                if not hid or hid in seen:
                    continue
                seen.add(hid)
                out.append(h)
                if len(out) >= fetch_k:
                    break
        except Exception as e:
            logger.error(f"INSPIRE fetch error for q={q!r}: {e}")
            continue
    return out


@mcp.tool()
async def search_topic_papers_semantic(
    topic: str,
    size: int = 20,
    fetch_k: int = 5,
    start_year: int = 0,
    end_year: int = 0,
    include_fulltext: bool = False,
    use_e5: bool = False,   # 默认用 MiniLM，避免第一次大下载
) -> TextContent:
    """
    语义匹配（大意相近）：先宽松召回，再用句向量余弦相似度重排。
    若缺少依赖/下载过慢，将优雅回退到基础检索。
    """
    import httpx, numpy as np

    phrase = topic.strip().replace('"', '\\"')
    date_clause = ""
    if start_year > 0 and end_year > 0:
        date_clause = f" and date:{start_year}->{end_year}"

    q_list = [
        f'(abstracts.value:"{phrase}" or t:"{phrase}" or k:"{phrase}"){date_clause}',
        f'(abstracts.value:{phrase} or t:{phrase} or k:{phrase}){date_clause}',
        f'"{phrase}"{date_clause}',
    ]
    if include_fulltext:
        q_list.insert(0, f'fulltext:"{phrase}"{date_clause}')

    fields = ",".join([
        "titles.title",
        "authors.full_name",
        "arxiv_eprints.value",
        "earliest_date",
        "citation_count",
        "abstracts.value",
    ])

    timeout = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)

    # —— 召回候选（小批量 + 明确超时，不做 429 退避） —— #
    async with httpx.AsyncClient(timeout=timeout, headers={"User-Agent": "inspire-topic-semantic/fast-1.0"}) as client:
        seen = set(); candidates = []
        per_page = min(fetch_k, 200)  # 保守，避免大载荷
        for q in q_list:
            if len(candidates) >= fetch_k:
                break
            try:
                resp = await client.get("https://inspirehep.net/api/literature",
                                        params={"q": q, "size": per_page, "fields": fields})
                if resp.status_code == 429:
                    # 直接返回提示，避免等待导致 MCP 超时
                    return TextContent(type="text", text="INSPIRE 触发限流（429）。请稍后再试或减少 fetch_k。")
                resp.raise_for_status()
                hits = resp.json().get("hits", {}).get("hits", [])
                for h in hits:
                    hid = h.get("id")
                    if not hid or hid in seen:
                        continue
                    seen.add(hid)
                    candidates.append(h)
                    if len(candidates) >= fetch_k:
                        break
            except Exception as e:
                # 网络异常时继续尝试下一种查询
                logger.warning(f"Recall error for q={q!r}: {e}")
                continue

    if not candidates:
        return TextContent(type="text", text=f"未召回到候选。请放宽 topic、增大 fetch_k 或开启 include_fulltext。")

    # —— 语义重排 —— #
    try:
        from sentence_transformers import SentenceTransformer
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        if use_e5:
            model_name = "intfloat/multilingual-e5-base"
        model = SentenceTransformer(model_name)

        q_text = f"query: {topic}" if use_e5 else topic

        docs, metas = [], []
        for h in candidates:
            md = h.get("metadata", {})
            title = (md.get("titles") or [{}])[0].get("title", "")
            abstract = (md.get("abstracts") or [{}])[0].get("value", "")
            if not title and not abstract:
                continue
            doc = f"{title}\n\n{abstract}".strip()
            if use_e5:
                doc = f"passage: {doc}"
            docs.append(doc); metas.append(md)

        if not docs:
            return TextContent(type="text", text="候选缺少可用摘要，无法做语义匹配。")

        q_emb = model.encode([q_text], normalize_embeddings=True)[0]
        d_emb = model.encode(docs, normalize_embeddings=True, batch_size=64)
        sims = (d_emb @ q_emb)

        import numpy as _np
        order = _np.argsort(-sims)[: max(1, min(size, len(metas)))]
        lines = [
            f'Semantic topic search: "{topic}"',
            f"(model: {model_name}, candidates: {len(docs)}, top-{size})",
            "=" * 80
        ]
        for rank, idx in enumerate(order, 1):
            md = metas[idx]
            sim = float(sims[idx])
            title = (md.get("titles") or [{}])[0].get("title", "Untitled")
            ed = md.get("earliest_date"); year = ed[:4] if isinstance(ed, str) and len(ed) >= 4 else "N/A"
            cites = md.get("citation_count", 0)
            arxiv_id = (md.get("arxiv_eprints") or [{}])[0].get("value", "N/A")
            arxiv_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id != "N/A" else "N/A"
            authors = [a.get("full_name", "") for a in (md.get("authors") or [])]
            authors_disp = (", ".join(authors[:3]) + " et al.") if len(authors) > 3 else (", ".join(authors) or "Unknown")
            abstract = (md.get("abstracts") or [{}])[0].get("value", "")
            snippet = (abstract[:260] + "…") if len(abstract) > 260 else abstract

            lines.append(
                f"{rank}. [{year}] {title}\n"
                f"   Score   : {sim:.3f}\n"
                f"   Authors : {authors_disp}\n"
                f"   Citations: {cites}\n"
                f"   arXiv   : {arxiv_id} → {arxiv_link}\n"
                f"   Abstract: {snippet}"
            )
        return TextContent(type="text", text="\n".join(lines))

    except ImportError:
        # 依赖缺失 → 回退
        return await search_topic_papers(topic=topic, size=size, start_year=start_year, end_year=end_year, include_fulltext=include_fulltext)
    except Exception as e:
        # 其他异常 → 回退基础检索并附错误信息
        base = await search_topic_papers(topic=topic, size=size, start_year=start_year, end_year=end_year, include_fulltext=include_fulltext)
        return TextContent(type="text", text=f"[语义重排失败：{type(e).__name__}: {e}]\n\n{base.text}")





# -----------------------------
# 主程序入口
# -----------------------------
def main():
    logger.info('Starting inspire-author-search MCP server')
    mcp.run('stdio')


if __name__ == "__main__":
    main()