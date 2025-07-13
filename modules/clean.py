from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString, SoupStrainer, Tag
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
MAX_TOKENS = 512  # ColBERT/BERT limit
ROW_DELIM = "▁CELL"
WIDE_COL_THRESHOLD = 30

PAGE_NUM_RE = re.compile(r"^\s*(?:F-\d+|\d+)\s*$", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^\s*(ITEM\s+\d+[A-Z]?|PART\s+[IVXLC]+)(?:\.\s*(.*))?$",
    re.IGNORECASE | re.MULTILINE,
)

ASCII_LINE = re.compile(r"( {2,}|\t).*\d")  # detect space-padded tables


def _collapse_ws(text: str) -> str:
    """Normalise Unicode, collapse runs of spaces/newlines."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _html_table_to_grid(table: Tag) -> list[list[str]]:
    """Expand rowspan/colspan and return a rectangular 2-D grid."""
    grid: list[list[str]] = []
    span: list[int] = []

    def esc(cell_text: str) -> str:
        return cell_text.replace("|", r"\|").replace("\n", " ")

    for tr in table.find_all("tr"):
        row: list[str] = []
        col = 0
        span = [max(0, x - 1) for x in span]  # decrement residual row-spans

        for cell in tr.find_all(["th", "td"]):
            while col < len(span) and span[col]:
                row.append("")
                col += 1

            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            txt = esc(cell.get_text(strip=True))

            for _ in range(cs):
                row.append(txt)
                span.append(rs)
                col += 1
        grid.append(row)

    width = max(map(len, grid))
    for r in grid:
        r.extend([""] * (width - len(r)))
    return grid


def _grid_to_markdown(grid: list[list[str]]) -> str:
    header = f"| {ROW_DELIM} | ".join(grid[0])
    sep = f"| {ROW_DELIM} | ".join(["---"] * len(grid[0]))
    body = ["| " + f" {ROW_DELIM} | ".join(row) + f" {ROW_DELIM} |" for row in grid[1:]]
    return "\n".join(
        ["| " + header + f" {ROW_DELIM} |", "| " + sep + f" {ROW_DELIM} |", *body]
    )


def _render_table_markdown(table: Tag) -> str:
    """Return Markdown for an HTML table, row-splitting if very wide."""
    grid = _html_table_to_grid(table)
    if len(grid[0]) > WIDE_COL_THRESHOLD:
        blocks = [_grid_to_markdown([grid[0], row]) for row in grid[1:]]
        return "\n\n".join(blocks)
    return _grid_to_markdown(grid)


def _ascii_block_to_md(lines: list[str]) -> str:
    """Convert space-padded (<pre>) table lines to Markdown."""
    cuts = sorted({m.start() for m in re.finditer(r"(?: {2,}|\t)", lines[0])})

    def split(line_: str) -> list[str]:
        segs, pos = [], 0
        for c in cuts:
            segs.append(line_[pos:c].strip())
            pos = c
        segs.append(line_[pos:].strip())
        return segs

    rows = list(map(split, lines))
    width = max(map(len, rows))
    for r in rows:
        r.extend([""] * (width - len(r)))

    header = f"| {ROW_DELIM} | ".join(rows[0])
    sep = f"| {ROW_DELIM} | ".join(["---"] * width)
    body = ["| " + f" {ROW_DELIM} | ".join(r) + f" {ROW_DELIM} |" for r in rows[1:]]
    return "\n".join(
        ["| " + header + f" {ROW_DELIM} |", "| " + sep + f" {ROW_DELIM} |", *body]
    )


STRAINER = SoupStrainer(
    name=lambda t: t
    in {"table", "tr", "td", "th", "p", "div", "pre", "li", "br", "ix:hidden"}
)


def html_to_markdown(html: str, safe_len: bool = True) -> str:
    """Convert SEC HTML to Markdown friendly for ColBERT chunking."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml", parse_only=STRAINER)

    for e in soup.select(
        "ix\\:hidden,[style*='display:none'],[style*='visibility:hidden']"
    ):
        e.decompose()

    for t in soup.find_all("table"):
        t.replace_with(NavigableString("\n\n" + _render_table_markdown(t) + "\n\n"))

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "li", "pre"]):
        blk.insert_before("\n")
        blk.insert_after("\n")

    text = soup.get_text()

    out: list[str] = []
    buf: list[str] = []

    for line in text.splitlines():
        if ASCII_LINE.search(line):
            buf.append(line)
            continue

        if buf:
            out.append(_ascii_block_to_md(buf) if len(buf) >= 3 else "\n".join(buf))
            buf.clear()

        if not PAGE_NUM_RE.match(line.strip()):
            out.append(line)

    if buf:
        out.append(_ascii_block_to_md(buf))

    md = "\n".join(out)

    def _promote(m: re.Match[str]) -> str:
        head = m.group(1).title()
        tail = (". " + m.group(2).strip().title()) if m.group(2) else ""
        return f"\n## {head}{tail}\n"

    md = re.sub(HEADING_RE, _promote, md)
    md = _collapse_ws(md)

    if safe_len and len(TOKENIZER(md)["input_ids"]) > MAX_TOKENS:
        mid = len(md) // 2
        return (
            html_to_markdown(md[:mid], safe_len)
            + "\n"
            + html_to_markdown(md[mid:], safe_len)
        )

    return md


def tokenize(text: str) -> list[str]:
    """A robust tokenizer that handles words with hyphens and apostrophes."""
    if not text:
        return []
    return re.findall(r"\w+(?:[-']\w+)*", text)
