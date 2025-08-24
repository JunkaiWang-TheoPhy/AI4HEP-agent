# server.py — 输出固定在“代码所在目录/latex_gen”，带依赖查找、TEXINPUTS 设置与编译工具
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
from pathlib import Path
from string import Template
import logging
import uuid
import subprocess
import shutil
import re
import os
from typing import List, Optional

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------
# FastMCP
# ------------------------------
mcp = FastMCP()

# ------------------------------
# Paths & Globals
# ------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()

def _resolve_default_output_dir() -> Path:
    """
    默认输出目录：
      1) 若提供 MCP_LATEX_OUTPUT_DIR 环境变量，则用它；
      2) 否则固定使用 <server.py 同级目录>/latex_gen
    """
    env = os.environ.get("MCP_LATEX_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (SCRIPT_DIR / "latex_gen").resolve()

OUTPUT_DIR: Path = _resolve_default_output_dir()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQ_STY = "jheppub.sty"
REQ_BST = "JHEP.bst"

JHEP_STY = OUTPUT_DIR / REQ_STY
JHEP_BST = OUTPUT_DIR / REQ_BST

def _refresh_paths():
    """当 output_dir 更新时，刷新依赖文件目标路径。"""
    global JHEP_STY, JHEP_BST
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JHEP_STY = OUTPUT_DIR / REQ_STY
    JHEP_BST = OUTPUT_DIR / REQ_BST

# ------------------------------
# Safety helpers
# ------------------------------
SAFE_TEX_RE = re.compile(r"^[\w.\- ]+\.tex$")
SAFE_AUXFILE_RE = re.compile(r"^[\w.\- ]+\.(sty|bst|bib|cls|cfg|def|clo|bbx|cbx|lbx|tex|log)$", re.IGNORECASE)
SAFE_BIB_RE = re.compile(r"^[\w.\- ]+\.bib$", re.IGNORECASE)

def _is_subpath(child: Path, parent: Path) -> bool:
    """Py3.8 兼容：判断 child 是否在 parent 之下。"""
    try:
        child = child.resolve()
        parent = parent.resolve()
        return str(child).startswith(str(parent) + os.sep) or child == parent
    except Exception:
        return False

def is_safe_tex_filename(name: str) -> bool:
    if not SAFE_TEX_RE.match(name):
        return False
    if "/" in name or "\\" in name:
        return False
    return _is_subpath((OUTPUT_DIR / name), OUTPUT_DIR)

def is_safe_aux_filename(name: str) -> bool:
    if not SAFE_AUXFILE_RE.match(name):
        return False
    if "/" in name or "\\" in name:
        return False
    return _is_subpath((OUTPUT_DIR / name), OUTPUT_DIR)

def is_safe_bib_filename(name: str) -> bool:
    if not SAFE_BIB_RE.match(name):
        return False
    if "/" in name or "\\" in name:
        return False
    return _is_subpath((OUTPUT_DIR / name), OUTPUT_DIR)

# ------------------------------
# Search & Copy helpers
# ------------------------------
def _split_path_list(env_val: Optional[str]) -> List[Path]:
    if not env_val:
        return []
    out = []
    seen = set()
    for raw in env_val.split(os.pathsep):
        s = raw.strip()
        if not s:
            continue
        p = Path(s).expanduser().resolve()
        if p.exists() and p.is_dir() and p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _candidate_search_dirs(user_dir: Optional[str] = None) -> List[Path]:
    """
    查找 jheppub.sty/JHEP.bst 的目录优先级：
      source_dir（若提供） → OUTPUT_DIR → SCRIPT_DIR → CWD → JHEP_TEX_DIR（可多路径）
    """
    dirs: List[Path] = []
    if user_dir:
        try:
            d = Path(user_dir).expanduser().resolve()
            if d.exists():
                dirs.append(d)
        except Exception:
            pass
    dirs += [OUTPUT_DIR, SCRIPT_DIR, Path.cwd().resolve()]
    dirs += _split_path_list(os.environ.get("JHEP_TEX_DIR", ""))
    # 去重
    uniq, seen = [], set()
    for d in dirs:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq

def _find_file_in_dirs(filename: str, search_dirs: List[Path]) -> Optional[Path]:
    for d in search_dirs:
        p = d / filename
        if p.exists() and p.is_file():
            return p.resolve()
    return None

def _copy_if_needed(src: Path, dst: Path) -> bool:
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(str(src), str(dst))
        return True
    return False

def ensure_jhep_dependencies(source_dir: Optional[str] = None) -> str:
    """确保 jheppub.sty 与 JHEP.bst 存在于 OUTPUT_DIR，必要时从候选目录复制。"""
    search_dirs = _candidate_search_dirs(source_dir)
    report, ok = [], True
    for fname in (REQ_STY, REQ_BST):
        dst = OUTPUT_DIR / fname
        if dst.exists():
            report.append(f"[OK] {fname} @ {dst}")
            continue
        src = _find_file_in_dirs(fname, search_dirs)
        if src:
            try:
                _copy_if_needed(src, dst)
                report.append(f"[COPY] {src} -> {dst}")
            except Exception as e:
                ok = False
                report.append(f"[ERROR] copy {fname}: {e}")
        else:
            ok = False
            places = ", ".join(str(d) for d in search_dirs)
            report.append(f"[MISSING] {fname} (searched: {places})")
    report.append("[DONE] deps ready." if ok else "[WARN] deps missing.")
    return "\n".join(report)

def _build_texinputs_env(extra_dirs: List[Path]) -> dict:
    """设置 TEXINPUTS，使 LaTeX 能在这些目录中搜索 .sty/.bst。"""
    env = os.environ.copy()
    items: List[Path] = []
    seen = set()
    for d in extra_dirs + _split_path_list(env.get("JHEP_TEX_DIR", "")):
        if d and d.exists() and d.is_dir() and d not in seen:
            items.append(d)
            seen.add(d)
    # 末尾空段表示继承系统默认 TEXMF 路径
    env["TEXINPUTS"] = os.pathsep.join(str(d) for d in items) + os.pathsep
    return env

# ------------------------------
# LaTeX Template
# ------------------------------
LATEX_TEMPLATE = Template(r"""% Auto-generated JHEP document
\documentclass[a4paper, 11pt]{article}
\usepackage{jheppub}
\usepackage{subcaption}
\usepackage{amsmath,amssymb,amscd,dsfont,enumerate,amsfonts,epsfig,mathtools,yfonts,bbold,breqn}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{xcolor}

% === Your Custom Commands ===
\def\d{\operatorname{d}}
\def\Dprod{\prod\limits_{i=1}^n\langle i,i+1\rangle}

\newcommand{\ca}{{\cal A}}\newcommand{\cc}{{\cal C}}\newcommand{\cd}{{\cal D}}
\newcommand{\ce}{{\cal E}}\newcommand{\ck}{{\cal K}}\newcommand{\cn}{{\cal N}}
\newcommand{\cm}{{\cal M}}\newcommand{\cl}{{\cal L}}\newcommand{\cf}{{\cal F}}
\newcommand{\ci}{{\cal I}}\newcommand{\cj}{{\cal J}}\newcommand{\cg}{{\cal G}}
\newcommand{\ch}{{\cal H}}\newcommand{\cv}{{\cal V}}\newcommand{\crr}{{\cal R}}
\newcommand{\co}{{\cal O}}\newcommand{\cp}{{\cal P}}\newcommand{\cs}{{\cal S}}
\newcommand{\cq}{{\cal Q}}\newcommand{\ct}{{\cal T}}\newcommand{\cu}{{\cal U}}
\newcommand{\cw}{{\cal W}}\newcommand{\cx}{{\cal X}}\newcommand{\cz}{{\cal Z}}
\newcommand{\nn}{{\nonumber}}
\def\bal#1\eal{{\begin{align}#1\end{align}}}
\def\alp[#1]{{\begin{align}#1\end{align}}}
\def\secnum[#1]{{\texorpdfstring{\(#1\)}{TEXT}}}
\def\sgn{{\text{sgn}}}
\def\eqa{{\begin{eqnarray}}}\def\eqae{{\end{eqnarray}}}
\def\eq{{\begin{equation}}}\def\eqe{{\end{equation}}}
\def\be{{\begin{equation}}}\def\ee{{\end{equation}}}
\def\bea{{\begin{eqnarray}}}\def\eea{{\end{eqnarray}}}
\def\ba{{\begin{array}}}\def\ea{{\end{array}}}
\def\bd{{\begin{displaymath}}}\def\ed{{\end{displaymath}}}
\def\ap{{ \alpha^{\prime} }}\def\eg{{ \it e.g.~ }}\def\ie{{ \it i.e.~ }}
\def\Tr{{ \rm Tr }}\def\tr{{ \rm tr }}\def\>{{ \rangle }}\def\<{{ \langle }}
\def\a{{ \alpha }}\def\b{{ \beta }}\def\c{{ \chi }}\def\del{{ \delta }}
\def\e{{ \epsilon }}\def\f{{ \phi }}\def\vf{{ \varphi }}\def\tvf{{ \tilde{ \varphi } }}
\def\g{{ \gamma }}\def\h{{ \eta }}\def\j{{ \psi }}\def\k{{ \kappa }}
\def\l{{ \lambda }}\def\m{{ \mu }}\def\n{{ \nu }}\def\w{{ \omega }}
\def\p{{ \pi }}\def\q{{ \theta }}\def\r{{ \rho }}\def\s{{ \sigma }}
\def\t{{ \tau }}\def\u{{ \upsilon }}\def\x{{ \xi }}\def\z{{ \zeta }}
\def\D{{ \Delta }}\def\F{{ \Phi }}\def\G{{ \Gamma }}\def\J{{ \Psi }}
\def\L{{ \Lambda }}\def\W{{ \Omega }}\def\P{{ \Pi }}\def\Q{{ \Theta }}
\def\S{{ \Sigma }}\def\U{{ \Upsilon }}\def\X{{ \Xi }}
\def\nab{{ \nabla }}\def\pa{{ \partial }}\def\da{{ \dot\alpha }}\def\db{{ \dot\beta }}

% Title, author, abstract
\title{$title}
\author[a]{$author}
\affiliation[a]{$affiliation}
\abstract{$abstract}

\begin{document}
\maketitle
\tableofcontents

$content

$bibblock
\end{document}
""")

# ------------------------------
# Tools
# ------------------------------
@mcp.tool()
def runtime_info() -> TextContent:
    """
    显示关键路径与环境变量（绝对路径）：
      - script_dir    : server.py 所在目录
      - cwd           : 进程当前工作目录
      - output_dir    : 实际写入/编译目录（默认 <script_dir>/latex_gen）
      - JHEP_TEX_DIR  : 依赖搜索额外目录（可多个，用系统分隔符）
      - MCP_LATEX_OUTPUT_DIR : 强制覆盖输出目录的环境变量
    """
    info = (
        f"script_dir : {SCRIPT_DIR}\n"
        f"cwd        : {Path.cwd().resolve()}\n"
        f"output_dir : {OUTPUT_DIR}\n"
        f"JHEP_TEX_DIR: {os.environ.get('JHEP_TEX_DIR','')}\n"
        f"MCP_LATEX_OUTPUT_DIR: {os.environ.get('MCP_LATEX_OUTPUT_DIR','')}\n"
    )
    return TextContent(type="text", text=info)

@mcp.tool()
def set_output_dir(path: str, create: bool = True, relative_to: str = "script") -> TextContent:
    """
    设置输出目录（随后所有 .tex/.pdf/.log 均写入该目录）。
    参数：
      - path: 目标路径；可相对路径或绝对路径
      - create: 是否自动创建（默认 True）
      - relative_to: 当 path 为相对路径时的基准：
          * "script": 相对 server.py 所在目录（默认，推荐）
          * "cwd"   : 相对当前工作目录
    说明：
      - 设置成功后会回显绝对路径；建议随后调用 runtime_info() 确认。
    """
    global OUTPUT_DIR
    try:
        base = SCRIPT_DIR if relative_to.lower() == "script" else Path.cwd().resolve()
        p = Path(path)
        new_dir = (base / p).resolve() if not p.is_absolute() else p.resolve()
        if create:
            new_dir.mkdir(parents=True, exist_ok=True)
        if not new_dir.exists():
            return TextContent(type="text", text=f"Path does not exist: {new_dir}")
        OUTPUT_DIR = new_dir
        _refresh_paths()
        return TextContent(type="text", text=f"output_dir set to: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"set_output_dir error: {e}")
        return TextContent(type="text", text=f"set_output_dir failed: {e}")

@mcp.tool()
def install_jhep_dependencies(source_dir: str = None) -> TextContent:
    """
    确保 jheppub.sty 与 JHEP.bst 存在于 output_dir（若缺失则从候选目录复制）。
    候选目录：source_dir → output_dir → script_dir → cwd → JHEP_TEX_DIR
    """
    try:
        report = ensure_jhep_dependencies(source_dir)
        return TextContent(type="text", text=report)
    except Exception as e:
        logger.error(f"install_jhep_dependencies error: {e}")
        return TextContent(type="text", text=f"install_jhep_dependencies failed: {e}")

@mcp.tool()
def write_bib_file(content: str, filename: str = "ref.bib", overwrite: bool = True) -> TextContent:
    """
    写入/覆盖 BibTeX 数据库（默认 ref.bib）到 output_dir。
    """
    if not is_safe_bib_filename(filename):
        return TextContent(type="text", text=f"Invalid .bib filename: {filename}")
    dst = (OUTPUT_DIR / filename).resolve()
    if dst.exists() and not overwrite:
        return TextContent(type="text", text=f"File exists and overwrite=False: {dst}")
    try:
        dst.write_text(content, encoding="utf-8")
        return TextContent(type="text", text=f"Bib file written: {dst}")
    except Exception as e:
        logger.error(f"write_bib_file error: {e}")
        return TextContent(type="text", text=f"write_bib_file failed: {e}")

@mcp.tool()
def create_latex_document(
    content: str,
    title: str = "Here is the title",
    author: str = "Here are authors",
    affiliation: str = "The affiliation",
    abstract: str = "This is the abstract"
) -> TextContent:
    """
    在 output_dir 下创建新的 .tex（JHEP 模板）。
    若 content 未提供 thebibliography/\\bibliography，则自动附加：
      \\bibliographystyle{JHEP}\n\\bibliography{ref}
    返回绝对路径与编译提示。
    """
    filename = f"jhep_doc_{uuid.uuid4().hex[:8]}.tex"
    filepath = OUTPUT_DIR / filename

    try:
        has_thebibliography = r"\begin{thebibliography}" in content
        has_bib_cmd = re.search(r"\\bibliography\s*{", content) is not None
        bibblock = "" if (has_thebibliography or has_bib_cmd) else "\\bibliographystyle{JHEP}\n \\bibliography{ref}"


        latex_content = LATEX_TEMPLATE.safe_substitute(
            title=title, author=author, affiliation=affiliation,
            abstract=abstract, content=content, bibblock=bibblock
        )

        filepath.write_text(latex_content, encoding='utf-8')

        msg = (
            f"✅ LaTeX created\n"
            f"  file      : {filepath.resolve()}\n"
            f"  output_dir: {OUTPUT_DIR}\n"
            f"  compile   : latexmk -pdf {filepath.name}  (or use compile_to_pdf)\n"
            f"  NOTE      : ensure {REQ_STY} & {REQ_BST} exist in output_dir (install_jhep_dependencies can help)\n"
        )
        return TextContent(type="text", text=msg)

    except Exception as e:
        logger.error(f"Failed to write LaTeX file: {e}")
        return TextContent(type="text", text=f"Failed to create LaTeX file: {e}")

@mcp.tool()
def append_to_document(filename: str, content: str, section_title: str = None) -> TextContent:
    """
    向 output_dir/filename 追加内容（优先插入到 \\bibliography 之前；否则 \\end{document} 之前；否则末尾）。
    """
    if not is_safe_tex_filename(filename):
        return TextContent(type="text", text=f"Invalid filename: {filename}")

    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        return TextContent(type="text", text=f"File not found: {filepath.resolve()}")

    try:
        lines = filepath.read_text(encoding='utf-8').splitlines(keepends=True)
        insert_pos = None
        for idx, line in enumerate(lines):
            if "\\bibliography" in line:
                insert_pos = idx
                break
        if insert_pos is None:
            for idx, line in enumerate(lines):
                if "\\end{document}" in line:
                    insert_pos = idx
                    break
        if insert_pos is None:
            insert_pos = len(lines)

        new_lines = []
        if section_title:
            new_lines.append(f"\n\\section{{{section_title}}}\n")
        new_lines.append(f"{content.strip()}\n\n")
        lines[insert_pos:insert_pos] = new_lines

        filepath.write_text("".join(lines), encoding='utf-8')
        return TextContent(type="text", text=f"Appended into: {filepath.resolve()}")

    except Exception as e:
        logger.error(f"Failed to append: {e}")
        return TextContent(type="text", text=f"Append failed: {e}")

def _find_artifact(stem: str, ext: str) -> Optional[Path]:
    """
    在 OUTPUT_DIR 里先精准查找 <stem>.<ext>，
    若不存在则递归查找（兼容 .latexmkrc/out_dir 等情况），
    返回按 mtime 最新的一个。
    """
    exact = OUTPUT_DIR / f"{stem}.{ext}"
    if exact.exists():
        return exact.resolve()
    candidates = list(OUTPUT_DIR.rglob(f"{stem}.{ext}"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0].resolve()
    return None

@mcp.tool()
def compile_to_pdf(filename: str) -> TextContent:
    """
    在 output_dir 内编译 .tex → .pdf；优先 latexmk，回退 pdflatex。
    - 强制 latexmk 输出到当前目录（-outdir=.）
    - 显式关闭把警告当错误（-Werror-），避免“有 PDF 却非零码”
    - 产物查找增加递归回退（_find_artifact）
    """
    if not is_safe_tex_filename(filename):
        return TextContent(type="text", text=f"Invalid filename: {filename}")

    filepath = (OUTPUT_DIR / filename).resolve()
    if not filepath.exists():
        return TextContent(type="text", text=f"File not found: {filepath}")

    dep_report = ensure_jhep_dependencies(None)

    has_latexmk = shutil.which("latexmk") is not None
    has_pdflatex = shutil.which("pdflatex") is not None
    has_bibtex = shutil.which("bibtex") is not None
    if not has_latexmk and not has_pdflatex:
        return TextContent(type="text", text="No LaTeX engine found (neither latexmk nor pdflatex in PATH).")

    log_path = OUTPUT_DIR / f"{filepath.stem}.compile.log"
    env = _build_texinputs_env([OUTPUT_DIR, SCRIPT_DIR])

    try:
        status_lines = []
        latexmk_rc = None
        if has_latexmk:
            cmd = [
                "latexmk",
                "-pdf",
                "-interaction=nonstopmode",
                "-file-line-error",
                "-outdir=.",       # 关键：强制产物到当前 OUTPUT_DIR
                "-Werror-",        # 关键：关闭把警告当错误（若不支持也无妨）
                filepath.name,
            ]
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(dep_report + "\n\n")
                p = subprocess.run(
                    cmd,
                    cwd=str(OUTPUT_DIR),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=env
                )
                latexmk_rc = p.returncode
                status_lines.append(f"latexmk exit={latexmk_rc}")

        else:
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(dep_report + "\n\n")
                p1 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-file-line-error", filepath.name],
                    cwd=str(OUTPUT_DIR), stdout=lf, stderr=subprocess.STDOUT, env=env
                )
                status_lines.append(f"Step 1: pdflatex → {'done' if p1.returncode == 0 else f'rc={p1.returncode}'}")

                need_bib = "\\bibliography" in filepath.read_text(encoding="utf-8", errors="ignore")
                aux_exists = (OUTPUT_DIR / f"{filepath.stem}.aux").exists()
                if need_bib and aux_exists and has_bibtex:
                    p2 = subprocess.run(
                        ["bibtex", filepath.stem],
                        cwd=str(OUTPUT_DIR), stdout=lf, stderr=subprocess.STDOUT, env=env
                    )
                    status_lines.append(f"Step 2: bibtex → {'done' if p2.returncode == 0 else f'rc={p2.returncode}'}")
                else:
                    reason = []
                    if not need_bib: reason.append("no \\bibliography")
                    if not aux_exists: reason.append("no .aux yet")
                    if need_bib and not has_bibtex: reason.append("bibtex not found")
                    status_lines.append(f"Step 2: bibtex skipped ({'; '.join(reason) or 'n/a'})")

                p3 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-file-line-error", filepath.name],
                    cwd=str(OUTPUT_DIR), stdout=lf, stderr=subprocess.STDOUT, env=env
                )
                p4 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-file-line-error", filepath.name],
                    cwd=str(OUTPUT_DIR), stdout=lf, stderr=subprocess.STDOUT, env=env
                )
                status_lines.append(f"Step 3: pdflatex x2 → {'done' if (p3.returncode==0 and p4.returncode==0) else 'failed'}")

        # —— 成功判定以“找到 PDF”为准 + 兼容递归查找 ——
        pdf_path = _find_artifact(filepath.stem, "pdf")

        if pdf_path:
            # latexmk 返回码非 0 的说明：通常为“仅有警告被视作错误”
            warn = ""
            if latexmk_rc is not None and latexmk_rc != 0:
                warn = " (note: latexmk exit != 0; likely warnings-as-errors, but PDF exists)"
            return TextContent(
                type="text",
                text=f"✅ PDF generated\n  pdf : {pdf_path}\n  log : {log_path.resolve()}\n  dir : {OUTPUT_DIR}\n" +
                     "\n".join(status_lines) + warn
            )
        else:
            return TextContent(
                type="text",
                text=f"❌ PDF not found\n  log : {log_path.resolve()}\n  dir : {OUTPUT_DIR}\n" + "\n".join(status_lines)
            )

    except Exception as e:
        logger.error(f"Compilation error: {e}")
        return TextContent(type="text", text=f"Compilation failed: {e}")


@mcp.tool()
def list_output_dir() -> TextContent:
    """
    列出 output_dir 的文件（绝对路径），便于核对“文件真的写到这里了”。
    """
    try:
        if not OUTPUT_DIR.exists():
            return TextContent(type="text", text=f"(missing) output_dir: {OUTPUT_DIR}")
        files = sorted(p.name for p in OUTPUT_DIR.iterdir() if p.is_file())
        listing = "\n".join(f"- {OUTPUT_DIR / name}" for name in files) if files else "(empty)"
        return TextContent(type="text", text=f"output_dir: {OUTPUT_DIR}\n{listing}")
    except Exception as e:
        logger.error(f"list_output_dir error: {e}")
        return TextContent(type="text", text=f"list_output_dir failed: {e}")

@mcp.tool()
def read_file(filename: str, max_chars: int = 4000) -> TextContent:
    """
    读取 output_dir/filename（.tex/.log/.bib/.sty/.bst 等）内容（最多 max_chars），用于排障。
    """
    if "/" in filename or "\\" in filename:
        return TextContent(type="text", text=f"Invalid filename: {filename}")
    p = (OUTPUT_DIR / filename).resolve()
    if not p.exists() or not p.is_file():
        return TextContent(type="text", text=f"File not found: {p}")
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + f"\n...\n[truncated to {max_chars} chars]"
        return TextContent(type="text", text=txt)
    except Exception as e:
        logger.error(f"read_file error: {e}")
        return TextContent(type="text", text=f"read_file failed: {e}")

# ------------------------------
# Startup info
# ------------------------------
def _startup_log():
    logger.info(f"[startup] script_dir = {SCRIPT_DIR}")
    logger.info(f"[startup] cwd        = {Path.cwd().resolve()}")
    logger.info(f"[startup] output_dir = {OUTPUT_DIR}")
    try:
        info = ensure_jhep_dependencies(None)
        for line in info.splitlines():
            if line.startswith("[MISSING]") or line.startswith("[WARN]"):
                logger.warning(line)
            else:
                logger.info(line)
    except Exception as e:
        logger.warning(f"Initial dependency check failed: {e}")

_startup_log()

# ------------------------------
# Main
# ------------------------------
def main():
    logger.info('Starting JHEP Latex-server')
    mcp.run('stdio')

if __name__ == "__main__":
    main() 