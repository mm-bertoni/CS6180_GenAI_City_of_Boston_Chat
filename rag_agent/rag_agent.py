"""RAG Agent:
   Reads the chroma_db/ collection built by notice-scraping.ipynb. Read only.
"""

import json
from asyncio import run
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# Should be same as the ingestion pipeline (notice-scraping.ipynb). 
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "public_notices"
LLM_MODEL = "gpt-4o-mini"

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_PATH = REPO_ROOT / "chroma_db"
API_KEY_PATH = REPO_ROOT / "open_ai_api_key.txt"

# Metadata fields that exist. 
ALLOWED_FILTERS = {"source_type", "cancelled", "notice_id", "public_testimony"}

EXTRACT_PROMPT = """You convert a user's question about Boston public notices into a search plan.

Available metadata fields for filtering:
- source_type: "pdf" or "page_text"
- cancelled: true or false
- public_testimony: true or false, whether the public can testify
- notice_id: a string of digits, e.g. "16492916"

Return ONLY valid JSON, no markdown fences, in this shape:
{{"search_text": "<the topical part of the question to search semantically>",
  "filters": {{"<field>": <value>}}}}

Use "filters" only for constraints that map to the fields listed above.
Remove from search_text any wording that you converted into a filter.
Anything about dates, times, or neighborhoods should stay in search_text for now.
If there are no applicable filters, use an empty object.

Question: {question}"""

ANSWER_PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say you don't know.
Cite the source number after each individual claim, not once at the end.
If a claim draws on multiple sources, cite all of them.

Each numbered block belongs to a specific notice. Never combine details from
different notices into one statement. If several notices match, list them
separately with their dates.

Context:
{context}

Question: {question}"""


def load_api_key(path=API_KEY_PATH):
    with open(path, encoding="utf-8-sig") as f:
        key = f.read().strip().strip('"').strip("'")
    if not key.startswith("sk-"):
        raise ValueError(f"Key doesn't look valid: {len(key)} chars starting {key[:6]!r}")
    return key


class RagAgent:
    """Answers questions from the public notices collection.

    answer() returns (answer_text, sources) where sources are plain dicts.
    """

    def __init__(self, k: int = 4):
        self.k = k

        self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self._vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=str(CHROMA_DB_PATH),
        )
        self._client = OpenAI(api_key=load_api_key())

    def chunk_count(self) -> int:
        """Sanity check"""
        return self._vectorstore._collection.count()

    def extract_search_plan(self, question: str):
        """Splitting question into (search_text, filters)."""
        resp = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(question=question)}],
            response_format={"type": "json_object"},  # guarantees parseable JSON
        )
        plan = json.loads(resp.choices[0].message.content)

        raw_filters = plan.get("filters") or {}
        filters = {k: v for k, v in raw_filters.items() if k in ALLOWED_FILTERS}

        # chroma wants none rather than when there is no filter
        return plan.get("search_text", question), filters or None

    @staticmethod
    def _to_source(doc) -> dict:
        """Converts LangChain Document into a plain dict.
        """
        m = doc.metadata
        return {
            "notice_id": m.get("notice_id"),
            "title": (m.get("title") or "").replace(" | Boston.gov", ""),
            "url": m.get("notice_url") or m.get("detail_url"),
            "source_type": m.get("source_type"),
            "file_label": m.get("file_label"),
            "event_datetime": m.get("event_datetime"),
            "text": doc.page_content,
        }

    async def answer(self, question: str):
        """Returns (answer_text, sources)."""
        search_text, filters = self.extract_search_plan(question)

        hits = self._vectorstore.similarity_search(search_text, k=self.k, filter=filters)

        # A filter matching nothing is not an error
        if not hits:
            return (
                "I couldn't find any public notices matching that. "
                f"(searched for {search_text!r} with filters {filters})"
            ), []

        sources = [self._to_source(d) for d in hits]

        # Numbering blocks and including title & event date
        context = "\n\n".join(
            f"[{i}] {s['title']} (notice {s['notice_id']}, "
            f"event {s['event_datetime']}, {s['source_type']})\n{s['text']}"
            for i, s in enumerate(sources, 1)
        )

        resp = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": ANSWER_PROMPT.format(
                context=context, question=question)}],
        )
        return resp.choices[0].message.content, sources


def format_sources(sources) -> str:
    """make sources as a numbered list.
    """
    if not sources:
        return ""
    lines = ["", "Sources:"]
    for i, s in enumerate(sources, 1):
        where = s["file_label"] or "notice webpage"
        lines.append(f" [{i}] [{s['source_type']}] {where} - {s['title']}")
        lines.append(f"     {s['url']}")
    return "\n".join(lines)


# For running interactively:
async def main():
    agent = RagAgent()
    print(f"{agent.chunk_count()} chunks loaded")
    question = input("Question about Boston public notices: ")
    answer_text, sources = await agent.answer(question)
    print(answer_text)
    print(format_sources(sources))


if __name__ == "__main__":
    run(main())
