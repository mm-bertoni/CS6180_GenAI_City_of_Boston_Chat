# CS6180_GenAI_Final_Project_City_of_Boston_RAG
Final project for NU CS6180 Foundations of Generative AI on RAG for City of Boston Public Notices

## Team Members
Margaret Bertoni \
Aidan Domondon \
Mohammad Aasim Shaikh \

## Project Description
We want to increase the accessibility of the City of Boston website to the general public while maintaining trust. The City of Boston's website currently has an AI Summary function, but it's citability is very limited, particularly when asking about public notices (for official meetings).
The public notices are stored as webpages, but also often attach PDFs with additional information to them.
Our project is a multi-agentic AI chatbot that uses RAG for questions about public notices/official meetings and a web search agent for other relevant city-government related questions. 

## File Descriptions
### Agents

### Public Notice Scraping
notice-scraping.ipynb: Jupyter notebook to scrape all of the current public notices (for future public meetings) \
archive-scraping.ipynb: Jupyter notebook to scrape the most recent 5% of archived public notices (subset selected due to large size of archive records) \ 
scraping_helpers.py: Common helper functions and variables for common use across current and archive notice scraping. \
public-notice-pdfs/: Folder where the scraped PDFs are saved as well as the scraping log file \
chroma_db/: Persistance folder for chroma_db \ 

## Tech Stack
Public Notice Scraping: requests, BeatifulSoup \
Embedding Model: HuggingFace: sentence-transformers/all-MiniLM-L6-v2 \
PDF Extraction: Docling \
RAG Documents: LangChain \
Vector Store: ChromaDB \


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

## Proposed Orchestrator Logging Schema
```
agent_log = {
 "conversation_id": one per user conversation,
 "attempt" attempt count - to track retries,
 "timestamp": timestamp allows us to measure latency,
 "source_type": "user" | "agent" | "tool"  (since Agents are treated like tools, may collapse agents into tools) 
 "source_name": agent/tool name,
 "input_message":,
 "routing_decision":,
 "target_type": "user" | "agent" | "tool" (since Agents are treated like tools, may collapse agents into tools),
 "target_name":,
 "output_message"
 "
}
```