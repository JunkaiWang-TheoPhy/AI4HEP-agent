# AI4HEP-agent

 

An MCP server exposing scholarly kinship tools powered by INSPIRE (literature/citations), arXiv (PDF access), and GROBID (PDF→TEI, in-text citations). Run via stdio and use with Claude Desktop or MCP Inspector.

## Features
- Fetch literature/author/citation metadata from **INSPIRE**; respect API limits (15 req / 5s).  
- Resolve arXiv IDs and download **PDF**; obey arXiv API etiquette (cache; add delays when paging).  
- Parse PDFs with **GROBID** `/api/processFulltextDocument` (optionally `segmentSentences=1`) to extract sections and in-text citations.  
- Compute **Kin(A,B)** = α·BibliographicCoupling + β·CoCitation + η·(φ_{A→B}+φ_{B→A})/2.  
- MCP tools: `pair_relation`, `citation_contexts`, `ingest_pdf`, `parse_pdf_tei`, `resolve_work`, `build_citation_graph`.

## Install
```bash
# using uv (recommended)
uv venv && source .venv/bin/activate
uv sync

# or pip
pip install -e .
