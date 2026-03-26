from unittest.mock import Mock

import pytest
from ebooklib import epub

from epub2anki.toc import extract_html

# --- Assume HTMLCleaner and extract_anchor_html are imported here ---


@pytest.fixture
def mock_book():
    """Fixture to provide a mocked EpubBook and EpubItem."""
    book = Mock(spec=epub.EpubBook)
    mock_item = Mock(spec=epub.EpubItem)
    book.get_item_with_href.return_value = mock_item
    book.mock_item = mock_item
    return book


def test_extract_full_file_adds_html_body_wrappers(mock_book):
    """If there are no anchors, the whole file is parsed, and wrapped if necessary."""
    html_content = "<p>Standard content</p>"
    mock_book.mock_item.get_content.return_value = html_content.encode("utf-8")

    result = extract_html(mock_book, "chapter1.html")
    assert result == "<html><body><p>Standard content</p></body></html>"


def test_extract_between_two_anchors(mock_book):
    """It successfully slices HTML from href1 up to href2."""
    html_content = (
        "<h1>Title</h1>"
        "<h2 id='start'>Section 1</h2>"
        "<p>Content 1</p>"
        "<h2 id='end'>Section 2</h2>"
        "<p>Content 2</p>"
    )
    mock_book.mock_item.get_content.return_value = html_content.encode("utf-8")

    result = extract_html(mock_book, "chapter1.html#start", "chapter1.html#end")

    # Notice <h2 id='end'> and everything after it is excluded
    expected = "<html><body><h2 id='start'>Section 1</h2><p>Content 1</p></body></html>"
    assert result == expected


def test_ignores_href2_if_different_file(mock_book):
    """If href2 is in a different file, it should slice to the end of the current file."""
    html_content = "<h1 id='start'>Start</h1><p>End of file 1</p>"
    mock_book.mock_item.get_content.return_value = html_content.encode("utf-8")

    # href2 points to chapter2.html, so it ignores it for chapter1.html
    result = extract_html(mock_book, "chapter1.html#start", "chapter2.html#target")

    expected = "<html><body><h1 id='start'>Start</h1><p>End of file 1</p></body></html>"
    assert result == expected


def test_drops_unpaired_opening_tags(mock_book):
    """If a tag is opened but never closed inside the slice, it is dropped."""
    html_content = (
        "<div class='wrapper'><h2 id='target'>Heading</h2><p>Content</p>"
        # Notice there is no </div> in the file (or it got cut off)
    )
    mock_book.mock_item.get_content.return_value = html_content.encode("utf-8")

    result = extract_html(mock_book, "chapter.html#target")

    # The <div class='wrapper'> is outside the anchor, so it's skipped.
    # We only get the valid paired tags from the anchor downwards.
    expected = "<html><body><h2 id='target'>Heading</h2><p>Content</p></body></html>"
    assert result == expected


def test_drops_unpaired_closing_tags(mock_book):
    """If a tag is closed but was never opened inside the slice, it is dropped."""
    html_content = (
        "<h2 id='target'>Heading</h2>"
        "<p>Content</p>"
        "</div></section>"  # Dangling closing tags from parent structures
    )
    mock_book.mock_item.get_content.return_value = html_content.encode("utf-8")

    result = extract_html(mock_book, "chapter.html#target")

    # The </div> and </section> are dropped because they have no matching opening tag in the slice
    expected = "<html><body><h2 id='target'>Heading</h2><p>Content</p></body></html>"
    assert result == expected


def test_preserves_void_elements(mock_book):
    """Self-closing/void elements (like <img>, <hr>, <br>) are preserved."""
    html_content = (
        "<h2 id='target'>Heading</h2>"
        "<img src='image.jpg' />"
        "<p>Some text<br>More text</p>"
        "<hr>"
    )
    mock_book.mock_item.get_content.return_value = html_content.encode("utf-8")

    result = extract_html(mock_book, "chapter.html#target")

    # Void elements shouldn't be stripped just because they don't have an explicit closing tag
    expected = "<html><body><h2 id='target'>Heading</h2><img src='image.jpg' /><p>Some text<br>More text</p><hr></body></html>"
    assert result == expected
