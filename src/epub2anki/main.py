"""
# Function

Automatic generation of Anki cards about important content in an epub book.

# Approach

1. Construct a representation of the TOC that maps title to chunk in the book under that item.
2. Generate a tree representation of the book with each subtree representing a TOC item.
3. Generate a sequence of subtrees that hosts text with a size that is close to a user-defined upper bound.
4. Reconstruct a flattened representation of each subtree.
5. Preprocess the flattened representation into an LLM-friendly markdown document.
6. Use an LLM to generate Anki cards for each markdown document.
7. Export the Anki cards to a file such that it is importable by an Anki application.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import genanki
from tqdm import tqdm

from .db import get_cached_notes, init_db, save_notes_to_cache
from .prompt_completion import RateLimiter, generate, generate_batch, retrieve_batch
from .prompt_generation import get_path_str, tree_to_prompt
from .toc import flatten, parse, prune

# ==========================================
# Helper Functions (Code Reuse)
# ==========================================


def build_chunk_mappings(book, trees, book_name, conn):
    """Finds cached notes and builds ID mappings for un-cached sections.

    Args:
        book (Book): The parsed book containing the Table of Contents tree.
        trees (list[SubTree]): A list of SubTree chunks generated from the TOC.
        book_name (str): The specific name of the book.
        conn (sqlite3.Connection): An active connection to the SQLite cache database.

    Returns:
        tuple[dict[str, str], dict[str, str], list[genanki.Note]]: A tuple containing:
            - prompts_to_batch: A dictionary mapping generated short IDs to LLM prompts.
            - id_to_path: A dictionary mapping generated short IDs to section paths.
            - cached_all: A list of all Genanki notes already present in the cache.
    """
    prompts_to_batch = {}
    id_to_path = {}
    cached_all = []

    for subtree in trees:
        section_path_str = get_path_str(subtree.path)
        cached_notes = get_cached_notes(conn, book_name, section_path_str)

        if cached_notes:
            cached_all.extend(cached_notes)
        else:
            prompt = tree_to_prompt(book, subtree)
            short_id = hashlib.md5(section_path_str.encode("utf-8")).hexdigest()
            prompts_to_batch[short_id] = prompt
            id_to_path[short_id] = section_path_str

    return prompts_to_batch, id_to_path, cached_all


def export_deck(all_notes, book_name, deck_id, output_dir):
    """Exports gathered Anki notes to an .apkg file.

    Args:
        all_notes (list[genanki.Note]): All notes generated or retrieved from cache.
        book_name (str): The name of the parsed book.
        deck_id (int): The unique integer ID for generating the Anki deck.
        output_dir (Path): The destination directory for the generated .apkg file.
    """
    if all_notes:
        deck_title = f"{book_name.replace('_', ' ').title()} Deck"
        my_deck = genanki.Deck(deck_id, deck_title)

        for note in all_notes:
            my_deck.add_note(note)

        output_file = output_dir / f"{book_name}.apkg"
        genanki.Package(my_deck).write_to_file(str(output_file))
        print(f"\nSuccess! Saved {len(all_notes)} notes to {output_file}")
    else:
        print("\nPipeline finished, but no notes were generated or found in cache.")


# ==========================================
# Main Application
# ==========================================


def main():
    """Parses command-line arguments and coordinates the generation of Anki flashcards from an EPUB book.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Generate Anki flashcards from EPUB books using an LLM."
    )
    parser.add_argument("book_path", type=Path, help="Path to the EPUB book.")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use Anthropic's async Batch API for 50%% lower costs.",
    )
    parser.add_argument(
        "--fetch-batch",
        type=str,
        help="Fetch an existing batch ID from Anthropic and build the deck.",
    )
    parser.add_argument(
        "--deck-id",
        type=int,
        default=2059400110,
        help="Unique integer ID for the Anki deck.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Maximum text size per LLM prompt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5",
        help="Anthropic model.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of times to retry failed LLM calls.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path.home() / ".cache" / "anki_cache.sqlite",
        help="Path to SQLite cache database.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("decks"),
        help="Directory to save the finished .apkg file.",
    )
    parser.add_argument(
        "--rate-max-requests",
        type=int,
        default=45,
        help="Maximum number of requests per rolling rate window.",
    )
    parser.add_argument(
        "--rate-max-input",
        type=int,
        default=45000,
        help="Maximum number of input tokens per rolling rate window.",
    )
    parser.add_argument(
        "--rate-max-output",
        type=int,
        default=9000,
        help="Maximum number of output tokens per rolling rate window.",
    )
    parser.add_argument(
        "--rate-window",
        type=int,
        default=60,
        help="Rolling rate window size in seconds.",
    )

    args = parser.parse_args()

    if args.book_path.suffix.lower() != ".epub":
        print(
            f"Error: The provided file '{args.book_path.name}' is not an EPUB file.",
            file=sys.stderr,
        )
        print("Please provide a valid .epub book.", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr
        )
        print("Please set your Anthropic API key to use this tool.", file=sys.stderr)
        print("Example: export ANTHROPIC_API_KEY='your-key-here'", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(args.db_path)
    book_name = args.book_path.stem

    print(f"Loading and parsing '{book_name}'...")
    book = parse(args.book_path, args.db_path)
    prune(
        book, {"Preface", "Glossary", "About the Authors", "Index", "Table of Contents"}
    )
    trees = flatten(book, args.chunk_size)

    # 1. Build context: Gather cached items and prepare map for missing items
    prompts_to_batch, id_to_path, all_notes = build_chunk_mappings(
        book, trees, book_name, conn
    )

    # 2. Execution Paths
    try:
        if args.fetch_batch:
            # --- FETCH EXISTING BATCH PATH ---
            print(f"Retrieving existing batch results for ID: {args.fetch_batch}...")
            batch_results = retrieve_batch(args.fetch_batch)

            for short_id, notes in batch_results.items():
                if short_id in id_to_path:
                    original_path = id_to_path[short_id]
                    prompt = prompts_to_batch[short_id]
                    save_notes_to_cache(
                        conn, book_name, original_path, prompt, args.model, notes
                    )
                    all_notes.extend(notes)
                else:
                    print(
                        f"Warning: Batch returned ID {short_id} which does not map to current book structure."
                    )

        elif args.batch:
            # --- SUBMIT NEW BATCH PATH ---
            if prompts_to_batch:
                print(
                    f"Found {len(prompts_to_batch)} un-cached chunks. Submitting batch..."
                )
                batch_id = generate_batch(prompts_to_batch, args.model)
                batch_results = retrieve_batch(batch_id)

                for short_id, notes in batch_results.items():
                    original_path = id_to_path[short_id]
                    prompt = prompts_to_batch[short_id]
                    save_notes_to_cache(
                        conn, book_name, original_path, prompt, args.model, notes
                    )
                    all_notes.extend(notes)
            else:
                print("All chunks were found in the cache! No batch submitted.")

        else:
            # --- UNBATCHED (SYNCHRONOUS) EXECUTION PATH ---
            if prompts_to_batch:
                limiter = RateLimiter(
                    max_requests=args.rate_max_requests,
                    max_input=args.rate_max_input,
                    max_output=args.rate_max_output,
                    window_seconds=args.rate_window,
                )

                print(f"Processing {len(prompts_to_batch)} sections synchronously...")
                for short_id, prompt in (
                    pbar := tqdm(prompts_to_batch.items(), desc="Generating Notes")
                ):
                    new_notes = generate(prompt, limiter, args.retries, args.model)
                    if new_notes:
                        original_path = id_to_path[short_id]
                        save_notes_to_cache(
                            conn,
                            book_name,
                            original_path,
                            prompt,
                            args.model,
                            new_notes,
                        )
                        all_notes.extend(new_notes)
                    pbar.set_postfix({"notes": len(all_notes)})
            else:
                print(
                    "All chunks were found in the cache! No synchronous requests made."
                )
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exporting notes generated so far...")
    finally:
        conn.close()

        # 3. Final Export
        export_deck(all_notes, book_name, args.deck_id, args.output_dir)


if __name__ == "__main__":
    main()
