<p align="center">
  <img src="assets/logo.png" width="300" alt="epub2anki logo">
</p>

# epub2anki

Convert EPUB ebooks into Anki flashcards using Anthropic's Claude API.

## Overview

`epub2anki` transforms your EPUB books into Anki decks (`.apkg` files). It parses the table of contents and internal book structure, divides the text into manageable chunks, and requests an LLM (Claude) to generate comprehensive and useful Anki flashcards.

## Features

- **Structural Parsing:** Uses the EPUB's Table of Contents to intelligently split the book into coherent sections.
- **LLM Flashcard Generation:** Uses Anthropic's API to construct high-quality flashcards summarizing key concepts.
- **Batch Processing:** Can utilize Anthropic's Batch API for up to 50% cost savings on API calls.
- **Resilient Coaching & Caching:** Uses SQLite to cache generated notes, meaning if the process is interrupted, you won't be charged twice for previously processed sections!
- **Direct Anki Export:** Outputs a ready-to-import `.apkg` file.

## Prerequisites

- Python 3.13+
- An [Anthropic API Key](https://console.anthropic.com/)

## Installation

You can install `epub2anki` using `pip` or `uv`:

```bash
pip install epub2anki
```

Or using `uv` (recommended):

```bash
uv tool install epub2anki
```

## Usage

Basic usage:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
epub2anki path/to/your/book.epub
```

Alternatively, you can pass the API key directly via the CLI:

```bash
epub2anki path/to/your/book.epub --api-key "your-api-key-here"
```

This will parse the EPUB, split it into chunks of ~50,000 characters, generate flashcards using the `claude-haiku-4-5` model, and finally save a `<book-name>.apkg` file in the current working directory.

### Advanced Usage & Batching

To save 50% on API costs, use the `--batch` flag. This will submit all generation requests to the Anthropic Batch API:

```bash
epub2anki path/to/your/book.epub --batch
```
*Note: The Batch API operates asynchronously and takes 5 minutes to 24 hours to finish. `epub2anki` will submit the batch and return a Batch ID.*

Once your batch is ready (you can check your Anthropic Console), run the script again using `--fetch-batch`:

```bash
epub2anki path/to/your/book.epub --fetch-batch msgbat_XXXXXXX
```
This will retrieve the completed responses from Anthropic, save them into the local cache, and generate your `.apkg` deck.

### Command-Line Arguments

```
usage: epub2anki [-h] [--batch] [--fetch-batch FETCH_BATCH] [--deck-id DECK_ID]
                 [--chunk-size CHUNK_SIZE] [--model MODEL] [--retries RETRIES]
                 [--api-key API_KEY] [--db-path DB_PATH] [--output-path OUTPUT_PATH]
                 [--rate-max-requests RATE_MAX_REQUESTS] [--rate-max-input RATE_MAX_INPUT]
                 [--rate-max-output RATE_MAX_OUTPUT] [--rate-window RATE_WINDOW]
                 book_path

Generate Anki flashcards from EPUB books using an LLM.

positional arguments:
  book_path             Path to the EPUB book.

options:
  -h, --help            show this help message and exit
  --batch               Use Anthropic's async Batch API for 50% lower costs.
  --fetch-batch ID      Fetch an existing batch ID from Anthropic and build the deck.
  --deck-id DECK_ID     Unique integer ID for the Anki deck.
  --chunk-size SIZE     Maximum text size per LLM prompt (default: 50000).
  --model MODEL         Anthropic model (default: claude-haiku-4-5).
  --api-key KEY         Anthropic API key (overrides ANTHROPIC_API_KEY env var).
  --output-path PATH    Path where the .apkg file should be saved (default: <cwd>/<book_name>.apkg).
```

## License

MIT License
