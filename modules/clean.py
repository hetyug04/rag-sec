import re

from bs4 import BeautifulSoup


def _render_table_to_markdown(table_element):
    """Converts a BeautifulSoup table element to a Markdown string."""
    markdown = ""
    # Extract headers from the first row if they exist
    headers = [
        th.get_text(strip=True).replace("\n", " ")
        for th in table_element.find_all("th")
    ]
    if headers:
        markdown += "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Extract data rows
    for row in table_element.find_all("tr"):
        cells = [
            td.get_text(strip=True).replace("\n", " ")
            for td in row.find_all(["td", "th"])
        ]
        if cells and cells != headers:  # Avoid duplicating header row
            markdown += "| " + " | ".join(cells) + " |\n"

    return "\n" + markdown + "\n"


def html_to_text(html_content: str) -> str:
    """
    Strips HTML tags from a filing and converts tables to Markdown
    to preserve their structure for the language model.
    """
    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "lxml")

    # Find all tables, convert them to Markdown, and replace the original table tag
    for table in soup.find_all("table"):
        markdown_table = _render_table_to_markdown(table)
        table.replace_with(soup.new_string(markdown_table))

    # Get the text from the modified soup, which now includes Markdown tables
    return soup.get_text(separator=" ", strip=True)


def tokenize(text: str) -> list[str]:
    """
    Splits a string of clean text into a list of words (tokens).
    """
    if not text:
        return []
    # Split on one or more whitespace characters
    return re.split(r"\s+", text)
