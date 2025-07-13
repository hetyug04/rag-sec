from __future__ import annotations

import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString, SoupStrainer, Tag

# --- Constants and Regexes ---
STRAINER = SoupStrainer(
    "p", "div", "pre", "table", "tr", "td", "th", "li", "br", "ix:hidden"
)
PAGE_NUM_RE = re.compile(r"^\s*(?:F-\d+|\d+)\s*$", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^\s*(ITEM\s+\d+[A-Z]?|PART\s+[IVXLC]+)(?:\.\s*(.*))?$",
    re.IGNORECASE | re.MULTILINE,
)

# --- Helper Functions ---


def _collapse_ws(text: str) -> str:
    """Normalise Unicode, collapse runs of spaces/newlines."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _html_table_to_grid(table: Tag) -> list[list[str]]:
    """
    Expand rowspan/colspan attributes to create a rectangular 2-D grid,
    perfectly representing the visual layout of the table.
    """
    grid: list[list[str]] = []

    for tr in table.find_all("tr"):
        for td in tr.find_all(["th", "td"]):
            # Get cell text and clean it
            txt = re.sub(r"\s+", " ", td.get_text(strip=True)).replace("|", r"\|")

            # Get rowspan and colspan attributes
            rowspan = int(td.get("rowspan", 1))
            colspan = int(td.get("colspan", 1))

            # Find the next available cell in the grid
            row_idx = len(grid)
            col_idx = 0
            if grid:
                # Find the first empty column in the current row
                while (
                    col_idx < len(grid[row_idx - 1])
                    and grid[row_idx - 1][col_idx] is not None
                ):
                    col_idx += 1

            # Place the cell text in the grid, expanding for colspan and rowspan
            for r in range(rowspan):
                for c in range(colspan):
                    # Ensure the grid is large enough
                    while len(grid) <= row_idx + r:
                        grid.append([None] * (col_idx + c + 1))
                    while len(grid[row_idx + r]) <= col_idx + c:
                        grid[row_idx + r].append(None)

                    grid[row_idx + r][col_idx + c] = txt

    # Replace all None placeholders with empty strings for consistent output
    return [["" if cell is None else cell for cell in row] for row in grid]


def _grid_to_markdown(grid: list[list[str]]) -> str:
    """Converts a 2D list of strings into a Markdown table."""
    if not grid:
        return ""

    # Ensure all rows have the same number of columns
    max_cols = max(len(row) for row in grid) if grid else 0
    for row in grid:
        row.extend([""] * (max_cols - len(row)))

    header = "| " + " | ".join(grid[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(grid[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in grid[1:]]
    return "\n".join([header, sep, *body])


# --- Core Function ---


def html_to_markdown(html: str) -> str:
    """Convert SEC HTML to Markdown friendly for ColBERT chunking."""
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml", parse_only=STRAINER)

    for e in soup.select(
        "ix\\:hidden,[style*='display:none'],[style*='visibility:hidden']"
    ):
        e.decompose()

    for t in soup.find_all("table"):
        grid = _html_table_to_grid(t)
        if grid:
            markdown_table = _grid_to_markdown(grid)
            t.replace_with(NavigableString("\n\n" + markdown_table + "\n\n"))
        else:
            t.decompose()

    for br in soup.find_all("br"):
        br.replace_with("\n")
    for blk in soup.find_all(["p", "div", "li", "pre"]):
        blk.insert_before("\n")
        blk.insert_after("\n")

    text = soup.get_text()

    out: list[str] = []
    for line in text.splitlines():
        if not PAGE_NUM_RE.match(line.strip()):
            out.append(line)

    md = "\n".join(out)

    def _promote(m: re.Match[str]) -> str:
        head = m.group(1).title()
        tail = (". " + m.group(2).strip().title()) if m.group(2) else ""
        return f"\n## {head}{tail}\n"

    md = re.sub(HEADING_RE, _promote, md)
    return _collapse_ws(md)
