import json
import sqlite3
from functools import cache
from pathlib import Path

import genanki

ANKI_MODEL_ID = 1847192314

SIMPLE_ANKI_MODEL = genanki.Model(
    ANKI_MODEL_ID,
    "Standard Model",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
    ],
    templates=[
        {
            "name": "Standard Card",
            "qfmt": "{{Front}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Back}}',
        },
    ],
)


@cache
def init_db(db_path: Path) -> sqlite3.Connection:
    """Initializes the SQLite database and creates required tables.
    
    Args:
        db_path (Path): The path to the SQLite database file.
        
    Returns:
        sqlite3.Connection: The established database connection.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_notes (
            book_name TEXT,
            section_path TEXT,
            prompt TEXT,
            notes_json TEXT,
            model TEXT,
            PRIMARY KEY (book_name, section_path)
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS subtrees (
            href TEXT PRIMARY KEY,
            html TEXT,
            size INTEGER
        )
    """
    )
    conn.commit()
    return conn


def get_cached_notes(
    conn: sqlite3.Connection, book_name: str, section_path: str
) -> list[genanki.Note]:
    """Retrieves generated Anki notes from the database cache for a specific book section.
    
    Args:
        conn (sqlite3.Connection): Active database connection.
        book_name (str): The name of the parsed book.
        section_path (str): The specific path of the section within the book.
        
    Returns:
        list[genanki.Note]: A list of retrieved Anki notes. Returns an empty list if no notes are cached.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT notes_json FROM generated_notes WHERE book_name = ? AND section_path = ?",
        (book_name, section_path),
    )
    row = cursor.fetchone()

    if not row:
        return []

    notes_data = json.loads(row[0])
    notes = []
    for data in notes_data:
        note = genanki.Note(
            model=SIMPLE_ANKI_MODEL,
            fields=[data["front"], data["back"]],
            tags=data["tags"],
        )
        notes.append(note)
    return notes


def save_notes_to_cache(
    conn: sqlite3.Connection,
    book_name: str,
    section_path: str,
    prompt: str,
    model: str,
    notes: list[genanki.Note],
):
    """Saves generated Anki notes to the database cache.
    
    Args:
        conn (sqlite3.Connection): Active database connection.
        book_name (str): The name of the parsed book.
        section_path (str): The specific path of the section within the book.
        prompt (str): The LLM prompt used for generation.
        model (str): The AI model name used for generation.
        notes (list[genanki.Note]): The list of generated notes to cache.
    """
    notes_data = [
        {"front": note.fields[0], "back": note.fields[1], "tags": note.tags}  # type: ignore
        for note in notes
    ]
    conn.execute(
        """
        INSERT OR REPLACE INTO generated_notes (book_name, section_path, prompt, model, notes_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (book_name, section_path, prompt, model, json.dumps(notes_data)),
    )
    conn.commit()
