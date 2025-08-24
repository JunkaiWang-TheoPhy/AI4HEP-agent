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

@mcp.tool()
async def search_author_papers(author_name: str) -> TextContent:
    """
    Search for an author on INSPIRE-HEP and return their publication list.

    Args:
        author_name: The full name or last name of the author (e.g., "Einstein", "Albert Einstein")

    Returns:
        A formatted text list of the author's papers, including title, year, citation count, and arXiv link.
    """
    logger.info(f"Searching for author: {author_name}")

    async with httpx.AsyncClient() as client:
        try:
            # Step 1: Search for the author by name
            search_url = "https://inspirehep.net/api/authors"
            params = {"q": author_name, "size": 5}
            response = await client.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data["hits"]["hits"]:
                return TextContent(type="text", text=f"No author found with name '{author_name}' on INSPIRE-HEP.")

            # Take the first match
            hit = data["hits"]["hits"][0]
            recid = hit["id"]
            display_name = hit["metadata"].get("name", {}).get("value", "Unknown")
            logger.info(f"Found author: {display_name} (recid: {recid})")

            # Step 2: Search for papers by this author
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

            # Format results
            result_lines = [f"Publications for author: {display_name}\n" + "=" * 60]
            for i, paper_hit in enumerate(papers_data["hits"]["hits"], 1):
                metadata = paper_hit["metadata"]
                title = metadata["titles"][0]["title"]

                # ✅ 安全获取年份
                year = "N/A"
                if "publication_info" in metadata and len(metadata["publication_info"]) > 0:
                    year = metadata["publication_info"][0].get("year", "N/A")
                elif "preprint_date" in metadata:
                    year = metadata["preprint_date"][:4]
                elif "earliest_date" in metadata:
                    year = metadata["earliest_date"][:4]

                citation_count = metadata.get("citation_count", 0)

                # ✅ 获取 arXiv 链接
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


# 主程序入口
def main():
    logger.info('Starting inspire-author-search MCP server')
    mcp.run('stdio')


if __name__ == "__main__":
    main()