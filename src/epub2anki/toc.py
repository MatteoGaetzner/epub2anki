import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Union
from warnings import warn

from ebooklib import epub
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from .db import init_db


class TOCNode(BaseModel):
    href: str
    next_href: str | None
    title: str
    size: int
    children: list["TOCNode"]


class Book(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    epub_book: epub.EpubBook
    children: list[TOCNode]


EbookLibTOCNode = tuple[epub.Section, list[Union["EbookLibTOCNode", epub.Link]]]


class SubTree(BaseModel):
    path: list[str]
    node: TOCNode


def parse(book_path: Path, db_path: Path, recompute: bool = False) -> Book:
    """Parses an EPUB book and computes its Table of Contents tree along with segment sizes.

    Args:
        book_path (Path): Path to the EPUB file.
        db_path (Path): Path to the SQLite cache database.
        recompute (bool, optional): Whether to force recomputation of sizes instead of using cache. Defaults to False.

    Returns:
        Book: The parsed hierarchical book representation.
    """
    print("Constructing TOC tree...")
    if not book_path.exists():
        raise FileNotFoundError(f"File {book_path} does not exist.")

    epub_book = epub.read_epub(book_path)
    conn = init_db(db_path)
    cursor = conn.cursor()

    # 2. Pre-compute the sequential reading order of hrefs to find boundaries
    ordered_hrefs = []

    def collect_hrefs(node: EbookLibTOCNode | epub.Link):
        """Recursively collects all hrefs from the EPUB's table of contents."""
        if isinstance(node, epub.Link):
            ordered_hrefs.append(node.href)
        else:
            section, children = node[0], node[1]
            ordered_hrefs.append(section.href)
            for child in children:
                collect_hrefs(child)

    for item in epub_book.toc:
        collect_hrefs(item)

    # Map each href to the one that immediately follows it
    next_href_map = {
        ordered_hrefs[i]: ordered_hrefs[i + 1] for i in range(len(ordered_hrefs) - 1)
    }

    def process_and_cache_node(href: str) -> int:
        """
        Extracts HTML between the current href and the next sequential href,
        calculates size, and caches it.
        """
        if not recompute:
            cursor.execute("SELECT size FROM subtrees WHERE href = ?", (href,))
            row = cursor.fetchone()
            if row is not None:
                return row[0]

        # Look up the boundary for this specific href
        href2 = next_href_map.get(href)

        html_text = extract_html(epub_book, href, href2)
        size = len(html_text.encode("utf-8"))

        cursor.execute(
            "INSERT OR REPLACE INTO subtrees (href, html, size) VALUES (?, ?, ?)",
            (href, html_text, size),
        )
        return size

    def dfs(node: EbookLibTOCNode | epub.Link) -> TOCNode:
        """Recursively traverses the EPUB's TOC to build the TOCNode tree."""
        if isinstance(node, epub.Link):
            size = process_and_cache_node(node.href)
            return TOCNode(
                href=node.href,
                next_href=next_href_map.get(node.href),
                title=node.title,
                size=size,
                children=[],
            )

        section, children = node[0], node[1]
        size = process_and_cache_node(section.href)

        children_out = []
        for c in children:
            child_node = dfs(c)
            children_out.append(child_node)
            size += child_node.size

        return TOCNode(
            href=section.href,
            next_href=next_href_map.get(section.href),
            title=section.title,
            size=size,
            children=children_out,
        )

    try:
        book = Book(epub_book=epub_book, children=[])
        for c in (pbar := tqdm(epub_book.toc)):
            book.children.append(dfs(c))
            # Safely get the title whether it's a Link or a tuple Section
            title = c[0].title if isinstance(c, tuple) else c.title
            title = title if len(title) <= 10 else title[:10] + "..."
            pbar.set_postfix_str(title)
    finally:
        conn.commit()

    return book


def prune(tok: Book, titles: set[str]) -> None:
    """Removes specific subtrees from the Book's Table of Contents by title.

    Args:
        tok (Book): The parsed book containing the Table of Contents tree.
        titles (set[str]): A set of titles to match against TOC nodes for removal.
    """

    def dfs(x: TOCNode):
        """Recursively prunes nodes from the TOC tree."""
        new_children = list()
        new_size = x.size
        for c in x.children:
            dfs(c)
            if c.title in titles:
                new_size -= c.size
                continue
            new_children.append(c)
        x.children = new_children
        x.size = new_size

    to_remove = list()
    for c in tok.children:
        if c.title in titles:
            to_remove.append(c)
            continue
        dfs(c)

    for c in to_remove:
        tok.children.remove(c)


def flatten(toc: Book, chunk_size: int = 50000, verbose: bool = False) -> list[SubTree]:
    """Flattens the Book's Table of Contents into a sequential list of SubTrees limited by chunk size.

    Args:
        toc (Book): The parsed book containing the TOC tree.
        chunk_size (int, optional): The maximum byte size per chunk. Defaults to 50000.
        verbose (bool, optional): If True, warns when a leaf exceeds the chunk size. Defaults to False.

    Returns:
        list[SubTree]: A list of aggregated subtrees.
    """
    out = list()

    def dfs(node: TOCNode, path: list[str]) -> None:
        cur_path = path + [node.title]
        if node.size <= chunk_size:
            out.append(SubTree(path=cur_path, node=node))
            return

        if verbose and not node.children and node.size > chunk_size:
            warn(
                f"Leaf '{'  -->  '.join(cur_path)}' has a size of {node.size} which is larger than chunk size {chunk_size}."
            )
            out.append(SubTree(path=cur_path, node=node))
            return

        tree = SubTree(
            path=cur_path,
            node=TOCNode(
                href=node.href,
                next_href=node.next_href,
                title=node.title,
                size=node.size - sum(c.size for c in node.children),
                children=[],
            ),
        )
        out.append(tree)

        for c in node.children:
            dfs(c, path + [node.title])

    for c in toc.children:
        dfs(c, [])
    return out


class HTMLCleaner(HTMLParser):
    """
    Parses a fragment of HTML, finds properly paired tags,
    and reconstructs the string while discarding unmatched dangling tags.
    """

    def __init__(self):
        """Initializes the HTMLCleaner with tracking structures for unpaired tags."""
        super().__init__()
        self.tokens = []
        self.void_elements = {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }

    def handle_starttag(self, tag, attrs):
        is_void = tag in self.void_elements
        self.tokens.append(
            {
                "type": "start",
                "tag": tag,
                "text": self.get_starttag_text(),
                "paired": is_void,  # Void elements don't need closing tags
            }
        )

    def handle_endtag(self, tag):
        self.tokens.append(
            {"type": "end", "tag": tag, "text": f"</{tag}>", "paired": False}
        )

    def handle_startendtag(self, tag, attrs):
        self.tokens.append(
            {
                "type": "startend",
                "tag": tag,
                "text": self.get_starttag_text(),
                "paired": True,
            }
        )

    def handle_data(self, data):
        self.tokens.append({"type": "data", "text": data, "paired": True})

    def handle_entityref(self, name):
        self.tokens.append({"type": "data", "text": f"&{name};", "paired": True})

    def handle_charref(self, name):
        self.tokens.append({"type": "data", "text": f"&#{name};", "paired": True})

    def get_clean_html(self) -> str:
        """Processes the tokens and returns the cleaned HTML string."""
        stack = []

        # Pass 1: Find properly paired opening and closing tags
        for i, token in enumerate(self.tokens):
            if token["type"] == "start" and not token["paired"]:
                stack.append(i)
            elif token["type"] == "end":
                # Traverse the stack backwards to find the matching start tag
                for j in range(len(stack) - 1, -1, -1):
                    start_idx = stack[j]
                    if self.tokens[start_idx]["tag"] == token["tag"]:
                        # Match found: mark both as paired
                        self.tokens[start_idx]["paired"] = True
                        token["paired"] = True
                        # Pop this tag and any unmatched tags opened after it
                        stack = stack[:j]
                        break

        # Pass 2: Reconstruct the string using only paired/valid tokens
        return "".join(token["text"] for token in self.tokens if token["paired"])


def extract_html(book: epub.EpubBook, href1: str, href2: str | None = None) -> str:
    """
    Extracts HTML between href1 and href2, strips unpaired tags,
    and ensures the result is validly wrapped in <html><body>...</body></html>.
    """
    if not href1:
        return ""

    parts1 = href1.split("#")
    base_href = parts1[0]
    anchor1 = parts1[1] if len(parts1) > 1 else None

    item = book.get_item_with_href(base_href)
    if item is None or not isinstance(item, epub.EpubItem):
        return ""

    content_bytes = item.get_content()
    try:
        html_text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        html_text = content_bytes.decode("latin-1")

    # 1. Find the start index
    start_idx = 0
    if anchor1:
        pattern1 = re.compile(
            rf"""(?:id|name)\s*=\s*['"]{re.escape(anchor1)}['"]""", re.IGNORECASE
        )
        match1 = pattern1.search(html_text)
        if match1:
            start_idx = html_text.rfind("<", 0, match1.start())
            if start_idx == -1:
                start_idx = match1.start()

    # 2. Find the end index (if href2 is provided and in the same file)
    end_idx = len(html_text)
    if href2:
        parts2 = href2.split("#")
        # Only search for the second anchor if it belongs to the current file.
        if parts2[0] == base_href and len(parts2) > 1:
            anchor2 = parts2[1]
            pattern2 = re.compile(
                rf"""(?:id|name)\s*=\s*['"]{re.escape(anchor2)}['"]""", re.IGNORECASE
            )
            match2 = pattern2.search(
                html_text, start_idx
            )  # Search strictly after start_idx
            if match2:
                end_idx = html_text.rfind("<", start_idx, match2.start())
                if end_idx == -1:
                    end_idx = match2.start()

    # 3. Slice the raw HTML
    raw_slice = html_text[start_idx:end_idx]

    # 4. Clean up unpaired/dangling tags
    cleaner = HTMLCleaner()
    cleaner.feed(raw_slice)
    cleaned_html = cleaner.get_clean_html()

    # 5. Ensure the document is structurally valid
    lower_html = cleaned_html.lower()
    if "<body" not in lower_html:
        cleaned_html = f"<body>{cleaned_html}</body>"
    if "<html" not in lower_html:
        cleaned_html = f"<html>{cleaned_html}</html>"

    return cleaned_html


def href_to_size(book: epub.EpubBook, href1: str, href2: str) -> int:
    """Calculates the byte size of HTML content between two href anchors in an EPUB book.

    Args:
        book (epub.EpubBook): The EPUB book object.
        href1 (str): The starting href anchor.
        href2 (str): The ending href anchor.

    Returns:
        int: Total size in bytes.
    """
    size = extract_html(book, href1, href2)
    return len(size.encode("utf-8"))
