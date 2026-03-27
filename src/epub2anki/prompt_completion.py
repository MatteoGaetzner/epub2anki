import itertools
import re
import sys
import time
from collections import deque
from typing import List

import genanki
import instructor
import markdown  # type: ignore
from anthropic import Anthropic
from pydantic import BaseModel, Field
from tqdm import tqdm

from .db import SIMPLE_ANKI_MODEL


class Flashcard(BaseModel):
    front: str = Field(..., description="The clear, self-contained question.")
    back: str = Field(
        ..., description="The concise answer, using markdown **bolding** for key terms."
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Contextual tags, e.g., ['Databases', 'Indexing']",
    )


class FlashcardList(BaseModel):
    cards: List[Flashcard]


class RateLimiter:
    def __init__(
        self,
        max_requests: int = 50,
        max_input: int = 25000,
        max_output: int = 6000,
        window_seconds: int = 60,
    ):
        """Initializes the RateLimiter with tracking queues and capacity limits.

        Args:
            max_requests (int): The maximum number of successful requests allowed within the window.
            max_input (int): The maximum number of input tokens allowed within the window.
            max_output (int): The maximum number of output tokens allowed within the window.
            window_seconds (int): The size of the rolling window in seconds.
        """
        # Stores tuples of: (timestamp, input_tokens, output_tokens)
        self.history: deque[tuple[float, int, int]] = deque()

        # Your specific limits
        self.MAX_REQUESTS = max_requests
        self.MAX_INPUT = max_input
        self.MAX_OUTPUT = max_output
        self.WINDOW_SECONDS = window_seconds

    def wait_for_capacity(self, estimated_input: int, estimated_output: int):
        """Blocks execution until the rolling window has enough capacity.

        Args:
            estimated_input (int): Expected input tokens for the upcoming request.
            estimated_output (int): Expected output tokens for the upcoming request.
        """
        while True:
            now = time.time()

            while self.history and now - self.history[0][0] > self.WINDOW_SECONDS:
                self.history.popleft()

            current_requests = len(self.history)
            current_input = sum(record[1] for record in self.history)
            current_output = sum(record[2] for record in self.history)

            if (
                current_requests < self.MAX_REQUESTS
                and (current_input + estimated_input) <= self.MAX_INPUT
                and (current_output + estimated_output) <= self.MAX_OUTPUT
            ):
                break

            time.sleep(1)

    def record_usage(self, actual_input: int, actual_output: int):
        """Called after the API returns to log the exact token usage.

        Args:
            actual_input (int): The actual input token count used in the request.
            actual_output (int): The actual output token count used in the request.
        """
        self.history.append((time.time(), actual_input, actual_output))


def generate(
    prompt: str,
    limiter: RateLimiter,
    max_retries: int = 3,
    model: str = "claude-haiku",
) -> list[genanki.Note]:
    """Generates Anki flashcards safely, retrying on failure up to max_retries.

    Args:
        prompt (str): The pre-formatted LLM prompt.
        limiter (RateLimiter): The rate limiter instance.
        max_retries (int): Maximum number of retries upon error.
        model (str): The model to query.

    Returns:
        list[genanki.Note]: Output notes. Empty list if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            notes = generate_unsafe(prompt, limiter, model)
            return notes
        except Exception as e:
            tqdm.write(f"Attempt {attempt + 1}: Error occurred: {e}. Retrying...")

    tqdm.write("Max retries reached. Skipping this section.")
    return []


def generate_unsafe(
    prompt: str, limiter: RateLimiter, model: str = "claude-haiku"
) -> List[genanki.Note]:
    """Takes a fully formatted prompt template and returns a list of valid genanki.Note objects.

    Args:
        prompt (str): The prompt indicating the section content.
        limiter (RateLimiter): Used to throttle outbound requests.
        model (str): Target model inference endpoint name.

    Returns:
        List[genanki.Note]: Converted Flashcard notes.
    """
    estimated_input = int(len(prompt.split()) * 1.3)
    estimated_output = 10  # Conservative estimate for a batch of flashcards

    limiter.wait_for_capacity(estimated_input, estimated_output)

    client = instructor.from_anthropic(Anthropic())

    extraction: FlashcardList = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        response_model=FlashcardList,
        extra_body={"cache_control": {"type": "ephemeral"}},
    )

    notes = []
    for card in extraction.cards:
        clean_tags = [tag.replace(" ", "_") for tag in card.tags]

        anki_front = markdown_to_anki_html(card.front)
        anki_back = markdown_to_anki_html(card.back)

        note = genanki.Note(
            model=SIMPLE_ANKI_MODEL, fields=[anki_front, anki_back], tags=clean_tags
        )
        notes.append(note)

    limiter.record_usage(estimated_input, estimated_output)
    return notes


def markdown_to_anki_html(text: str) -> str:
    """Safely converts Markdown to HTML while protecting MathJax elements from the parser.

    Args:
        text (str): Incoming markdown string with standard MathJax markup.

    Returns:
        str: HTML string containing Anki-compatible math delimiters.
    """
    math_blocks = []

    def repl_display(match):
        math_blocks.append(r"\[" + match.group(1) + r"\]")
        return f"MATHBLOCKREPLACEMENT{len(math_blocks)-1}"

    text = re.sub(r"\$\$(.*?)\$\$", repl_display, text, flags=re.DOTALL)

    def repl_inline(match):
        math_blocks.append(r"\(" + match.group(1) + r"\)")
        return f"MATHBLOCKREPLACEMENT{len(math_blocks)-1}"

    # Uses negative lookbehinds/lookaheads to ensure we don't accidentally match $$
    text = re.sub(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", repl_inline, text)

    html = markdown.markdown(text)

    for i, block in enumerate(math_blocks):
        html = html.replace(f"MATHBLOCKREPLACEMENT{i}", block)

    return html


# ==========================================
# Batch Functions
# ==========================================


def generate_batch(
    prompts_by_id: dict[str, str], model_name: str = "claude-haiku"
) -> str:
    """Takes a dictionary of {chunk_id: prompt} and submits them all at once to Anthropic.

    Args:
        prompts_by_id (dict[str, str]): A mapping from ID string to prompt text.
        model_name (str): Model parameter for the batch request.

    Returns:
        str: Extracted batch ID from the Anthropic API structure.
    """
    client = Anthropic()

    # Convert our Pydantic model into an Anthropic Tool Schema
    anki_tool = {
        "name": "generate_flashcards",
        "description": "Extract the core concepts into a structured list of flashcards.",
        "input_schema": FlashcardList.model_json_schema(),
    }

    requests = []
    for chunk_id, prompt in prompts_by_id.items():
        requests.append(
            {
                "custom_id": chunk_id,  # We use the tree path/chunk ID so we can map results back later
                "params": {
                    "model": model_name,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [anki_tool],
                    # Force the model to output using our JSON schema
                    "tool_choice": {"type": "tool", "name": "generate_flashcards"},
                },
            }
        )

    print(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted successfully! Batch ID: {batch.id}")

    return batch.id


def retrieve_batch(batch_id: str) -> dict[str, list[genanki.Note]]:
    """Polls the batch until complete, then parses results into Genanki objects.

    Args:
        batch_id (str): Existing remote Batch ID tracking string.

    Returns:
        dict[str, list[genanki.Note]]: Dictionary mapping chunk ID to list of Genanki notes.
    """
    client = Anthropic()
    print(f"\nTracking Batch ID: {batch_id}")
    print("If this script crashes, view/download results manually at:")
    print("👉 https://console.anthropic.com/settings/workspaces/default/batches\n")

    # A simple spinner using standard library
    spinner = itertools.cycle(["-", "\\", "|", "/"])
    poll_interval = 60

    while True:
        batch_status = client.messages.batches.retrieve(batch_id)
        status = batch_status.processing_status

        if status == "ended":
            # Clear the line one last time before printing the success message
            print("\r" + " " * 80 + "\r\nBatch processing complete!")
            break
        elif status in ["canceling", "canceled", "errored"]:
            print(f"\nBatch failed or was canceled. Final Status: {status}")
            return {}

        succeeded = batch_status.request_counts.succeeded
        processing = batch_status.request_counts.processing
        errored = batch_status.request_counts.errored
        canceled = batch_status.request_counts.canceled
        expired = batch_status.request_counts.expired
        total = succeeded + processing + errored + canceled + expired

        # Inner loop to update the CLI every second without hitting the API
        for remaining_seconds in range(poll_interval, 0, -1):
            spin_char = next(spinner)
            # \r returns cursor to the start of the line. flush=True forces the terminal to draw it immediately.
            status_text = f"\r{spin_char} Status: {status} (Processed {succeeded} / {total})... polling again in {remaining_seconds}s  "
            sys.stdout.write(status_text)
            sys.stdout.flush()
            time.sleep(1)

    results = {}
    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            notes = []
            for content_block in item.result.message.content:
                if (
                    content_block.type == "tool_use"
                    and content_block.name == "generate_flashcards"
                ):
                    try:
                        extraction = FlashcardList(**content_block.input)  # type: ignore
                        for card in extraction.cards:
                            clean_tags = [tag.replace(" ", "_") for tag in card.tags]
                            anki_front = markdown_to_anki_html(card.front)
                            anki_back = markdown_to_anki_html(card.back)

                            note = genanki.Note(
                                model=SIMPLE_ANKI_MODEL,
                                fields=[anki_front, anki_back],
                                tags=clean_tags,
                            )
                            notes.append(note)
                    except Exception as e:
                        print(f"Validation error on chunk {item.custom_id}: {e}")
            results[item.custom_id] = notes
        else:
            print(f"Chunk {item.custom_id} failed: {item.result.type}")

    return results
