from .toc import Book, SubTree, TOCNode, extract_html

prompt_template = """You are an expert educator and Anki flashcard creator. Your objective is to extract core concepts, definitions, mechanisms, and key facts from the provided text and convert them into high-quality Anki flashcards.

# Card Creation Principles
1. **Atomicity:** Each card must test exactly ONE concept. Break complex ideas (like lists or multi-part processes) into separate cards.
2. **Self-Contained Context:** The front of the card must be fully understandable in isolation. Never use pronouns (e.g., "it", "this theorem"). Explicitly name the subject, concept, or algorithm.
3. **Comprehension Over Trivia:** Focus on "why" and "how" rather than rote, disconnected memorization.
4. **Brevity & Emphasis:** Keep the back of the card extremely concise. Use Markdown **bolding** to highlight the 1-3 most critical keywords. If an answer requires listing items, you MUST place each item on a new line (e.g., `1. First\n2. Second`) rather than writing them inline, so markdown lists render correctly.
5. **Universal Knowledge:** The cards must test the underlying subject matter, NOT the book or document itself. Write the cards as if they belong in a generalized, universal knowledge base for this field of study.

# Strict Exclusions (What NOT to do)
- **NO Meta-References:** Never use phrases like "In Chapter 1", "According to the book", "As introduced in this section", "The author states", or explicit text structural references like "(Equation 1.6)" or "Figure 3".
- **NO Syllabus/Structural Questions:** Never create cards about the structure or outline of the text (e.g., "What are the four topics covered in this chapter?").
- **NO Source-Specific Trivia:** Skip introductory fluff, personal anecdotes from the author, or generalized context that lacks a specific, testable fact.
- **NO Example-Specific Cards:** If the section is a summary or review that references specific examples, case studies, or scenarios from earlier in the book (e.g., "the database example from Chapter 3", "the company discussed earlier"), do NOT create cards about those examples. Only create cards if the section presents new, generalizable concepts or principles that stand alone without requiring knowledge of book-specific examples.

# Examples
**Bad Card (Violates Atomicity and Focuses on Lists)**
- Front: What are the four properties of an ACID database transaction?
- Back: Atomicity, Consistency, Isolation, and Durability.

**Bad Card (Violates Universal Knowledge / Meta-Reference)**
- Front: As discussed in Section 4.2, what is the second step of cellular respiration?
- Back: The Krebs cycle.

**Bad Card (Example-Specific / Requires Book Context)**
- Front: In the retail company example, what was the primary reason for implementing a NoSQL database?
- Back: To handle the high volume of unstructured customer data.

**Good Card (Clear context, universal, testing mechanisms - Computer Science)**
- Front: In the context of database transactions, what specific guarantee does **Isolation** (the 'I' in ACID) provide?
- Back: It ensures that concurrent transactions execute as if they were running **sequentially**, preventing intermediate states from being visible to other operations.

**Good Card (Clear context, rigorous definitions - Mathematics)**
- Front: Let $V$ and $W$ be vector spaces over a field $F$, and let $T: V \\to W$ be a linear transformation. What is the definition of the **kernel** (or null space) of $T$?
- Back: It is the set of all vectors $v \\in V$ such that $T(v) = 0_W$, where $0_W$ is the **zero vector** in $W$.

**Good Card (Testing "why" and mechanisms - Biology)**
- Front: During the **Krebs cycle** (citric acid cycle), what is the primary biological purpose of oxidizing acetyl-CoA?
- Back: To extract high-energy electrons and transfer them to carrier molecules (**NADH** and **FADH₂**) for use in the electron transport chain.

# Execution Instructions

Before creating any cards, use your scratchpad to analyze the content:

<scratchpad>
First, analyze the section content:
1. Is this a summary, review, or concluding section?
2. Does it reference specific examples, case studies, or scenarios from earlier in the book?
3. Does it present new, generalizable concepts that stand alone, or does it merely recap book-specific content?
4. Identify which concepts are truly universal and testable without book context.
5. List the distinct, atomic concepts that warrant flashcard creation.
6. Verify that each concept can be understood and tested independently of the source material.
</scratchpad>

Then proceed to create cards:
1. Draft atomic, self-contained questions for the front of the cards.
2. Draft concise, bold-emphasized answers for the back of the cards.
3. Verify that no card references the book, author, chapter, or book-specific examples.
4. Format the final output.

# Output Format
Output ONLY valid JSON. Do not include introductory text, explanations, or markdown formatting blocks. The JSON must adhere exactly to this structure:

{{
  "cards": [
    {{
      "front": "The clear, self-contained question explicitly naming the concept.",
      "back": "The concise answer. Use **bolding** for key terms. For math, use strictly valid LaTeX ($inline$ or $$display$$) with **no HTML/Markdown tags inside the math environment**.",
      "tags": ["TopicTag1", "TopicTag2"]
    }}
  ]
}}

If no suitable cards can be created from the section (e.g., it only contains book-specific examples or structural content), output:

{{
  "cards": []
}}

## Tagging Rules
- Tags MUST represent the core topical domain (e.g., "Linear-Algebra", "Cell-Biology", "Database-Systems").
- NEVER use structural tags like "Chapter-1", "Overview", or "Philosophy".
- If the extracted concept is too general or foundational to warrant a specific domain tag, output an empty list: [].

# Context & Inputs

You will be provided with three pieces of information to help you create flashcards:

<TableOfContents>
{TOC}
</TableOfContents>

<Path>
{PATH}
</Path>

<SectionContent>
{SECTION}
</SectionContent>

Use the TableOfContents and Path strictly to understand the broader context and subject domain. Generate flashcards EXCLUSIVELY based on the facts explicitly presented in the SectionContent. Do not infer, hallucinate, or add outside information. If the SectionContent only references examples or case studies from elsewhere in the book without presenting new generalizable concepts, create no cards.

Begin your analysis in scratchpad tags, then output your final JSON response."""


def get_toc_str(book: Book) -> str:
    """Generates a formatted string representation of the book's table of contents.

    Args:
        book (Book): The parsed book containing the Table of Contents tree.

    Returns:
        str: A multi-line string where each line represents a node in the TOC, indented by its depth.
    """
    out = ""

    def dfs(node: TOCNode, depth: int) -> None:
        nonlocal out
        if out:
            out = out + "\n"
        out = out + ("    " * depth) + node.title
        for c in node.children:
            dfs(c, depth + 1)

    for c in book.children:
        dfs(c, 0)
    return out


def get_path_str(path: list[str]) -> str:
    """Formats a list of path strings into a breadcrumb-style string.

    Args:
        path (list[str]): The list of path components leading to the current section.

    Returns:
        str: A breadcrumb string representing the hierarchy.
    """
    return "root  ->  " + "  ->  ".join(map(lambda x: f"'{x}'", path))


def tree_to_prompt(book: Book, tree: SubTree) -> str:
    """Constructs the LLM prompt for a subtree by combining the TOC, path, and HTML content.

    Args:
        book (Book): The parsed book containing the Table of Contents tree.
        tree (SubTree): The specific subtree representing the current section.

    Returns:
        str: The fully constructed prompt for the LLM.
    """
    html = extract_html(book.epub_book, tree.node.href, tree.node.next_href)
    toc_str = get_toc_str(book)
    path_str = get_path_str(tree.path)
    return prompt_template.format(TOC=toc_str, PATH=path_str, SECTION=html)
