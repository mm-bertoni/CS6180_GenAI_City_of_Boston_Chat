◊# CS6180_GenAI_Final_Project_City_of_Boston_Chat
Final project for NU CS6180 Foundations of Generative AI a Multi-Agentic Chatbot for the  City of Boston with RAG on Public Notices

## Team Members
Margaret Bertoni \
Aidan Domondon \
Mohammad Aasim Shaikh 

## Project Description

City government websites can be difficult to navigate, particularly if a user isn’t familiar with the various departments and resources available. 
Allowing citizens to interact with the website using natural language would lower some of those barriers. \

The public notices are stored as webpages, but also often attach PDFs with additional information to them, so we wanted to use RAG to ensure that information from the PDFs is accessible. \

Our project is a multi-agentic AI chatbot that uses RAG for questions about public notices/official meetings and a web search agent for other relevant city-government related questions. 

## File Descriptions
### Agents
rag_agent/rag_agent.py: RAG Agent
orchestrator_agent.py: Web Search Agent, Answer Agent, Guardrail Agent and Orchestrator Agent with access to RAG Agent
orchestrator_agent_no_rag.py: Web Search Agent, Answer Agent, Guardrail Agent and Orchestrator Agent with NO RAG Agent (The Baseline for comparison)
### Public Notice Scraping
notice-scraping.ipynb: Jupyter notebook to scrape all of the current public notices (for future public meetings) \
archive-scraping.ipynb: Jupyter notebook to scrape the most recent 5% of archived public notices (subset selected due to large size of archive records) \ 
scraping_helpers.py: Common helper functions and variables for common use across current and archive notice scraping. \
public-notice-pdfs/: Folder where the scraped PDFs are saved as well as the scraping log file \
chroma_db/: Persistance folder for chroma_db 

### User Interface
ui/app.py: Streamlit chat app. Page layout, chat history, source cards, badges. Knows nothing about agents. \
ui/backend.py: The only file that touches the agents. Builds MultiAgent once, wraps its async call, returns (answer, sources, trace). \
ui/formatting.py: Pure functions - UTC to Boston time, citation parsing, grouping sources by notice. No Streamlit, no I/O. \
ui/styles.py: Optional CSS. Deleting the import and the inject_styles() call removes it cleanly. \
ui/tests/: Unit tests for formatting.py. Run with `python -m pytest ui/tests/ -q` \
.streamlit/config.toml: Theme (light and dark) plus `fileWatcherType = "none"`, which is required - torch breaks Streamlit's file watcher. 

### Evaluation/Testing
eval_testing.ipynb: Runs 21 test queries on the MultiAgent() class (with RAG access) and compares the resulting sources and tool_calls to the expecation. Also calculates latency (across the entire test, and then segmented by tool_call) \
eval_testing_baseline.ipynb: Runs 21 test queries on the MultiAgentNoRAG() class (baseline with No RAG) and compares the resulting sources and tool_calls to the expecation. Also calculates latency (across the entire test, and then segmented by tool_call) \
test_cases.py: Test cases for the evaluation suite
eval_suite_multiagent.json: Stores the raw logs for the evaluation suite on MultiAgent() (with RAG)\
eval_suite_no_rag.json: Stores the raw logs for the evaluation suite on MultiAgentNoRAG() (Baseline without RAG)


## Tech Stack
Public Notice Scraping: requests, BeatifulSoup \
Embedding Model: HuggingFace: sentence-transformers/all-MiniLM-L6-v2 \
PDF Extraction: Docling \
RAG Documents: LangChain \
Vector Store: ChromaDB \
Agent Framework: OpenAI Agents SDK \
User Interface: Streamlit 

## Running the UI

**Prerequisites**
- `open_ai_api_key.txt` in the repo root, containing the key
- A populated `chroma_db/`. It is loaded in the repo, but if there are any issues loading the db, can run `notice-scraping.ipynb` and then `archive-scraping.ipynb` to freshly populate the db.

**Run it**
```
venv/Scripts/python.exe -m streamlit run ui/app.py
```

The first launch takes about 30 seconds while the embedding model loads. That is
cached afterwards, so later questions only pay for the agents themselves
(roughly 10-15s for a notice question, longer if it falls back to web search).

## RAG Schema
The page texts and PDFs are separately added as docs to the ChromaDB. 

**The metadata for each LangChain doc:**
```
notice_metadata = {
    "notice_id": ,
    "title":,
    "cancelled":,
    public_testimony",
    "notice_url":,
    "posted_at":,
    "event_datetime":,
    "address_1":,
    "address_2":,
    "status":,
    "checked_at":,
}
```

**Metadata for PDFs**
```
"file_label": file name,
"file_hash":,
"source_type": "pdf"
```

**Metadata for Page Text:**
```
"source_type": "page_text",
"text_hash":,
```


### RAGAgent output shape

`RAGAgent.answer(query)` returns a 2-tuple: `(answer_text, sources)`.

- `answer_text` — `str`, the written answer
- `sources` — `list[dict]`, one dict per retrieved chunk (up to 4)

Each source dict:

| field | type | notes |
|-------|------|-------|
| `notice_id` | `str` | e.g. `"16492911"` |
| `title` | `str` |  |
| `url` | `str` | link to the notice page |
| `source_type` | `str` | `"pdf"` or `"page_text"` |
| `file_label` | `str` or `None` | PDF name; **`None` when `source_type` is `page_text`** |
| `event_datetime` | `str` | e.g. `"2026-07-22T13:00:00Z"` |
| `score` | `float` | similarity distance, **lower is better** |
| `text` | `str` | the chunk text that was retrieved |

Example:

```python
(
  "The Boston Retirement Board Meeting is scheduled for July 22, 2026, at 9:00 a.m. ...",
  [
    {
      "notice_id": "16492911",
      "title": "Boston Retirement Board Meeting",
      "url": "https://www.boston.gov/public-notices/16492911",
      "source_type": "page_text",
      "file_label": None,
      "event_datetime": "2026-07-22T13:00:00Z",
      "score": 0.4602,
      "text": "Boston Retirement Board Meeting | event 2026-07-22T13:00:00Z\nAdministrative Session"
    },
    ...
  ]
)
```

## Logging Schema
Because we are using the OpenAI Agents SDK, we were able to inspect traces through the OpenAI SDK web portal. To be able to inspect them locally (and evaluate the behavior of the system using our test cases), we created some lightweight logs:
```
trace_info = {
    "question":user_query,
    "tool_calls":,
    "guardrail_tripped":True/False,
    "guardrail_reason":,
    "tools_called":[rag | web_search],
    "sources",
    "latency_ms":latency in ms
}
```
It should be noted that the latency is measuring the orchestrator latency,  not the latency of the answer agent (which is only responsible for reformatting the information found and providing it to the user)