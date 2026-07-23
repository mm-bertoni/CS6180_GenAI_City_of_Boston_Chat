# CS6180_GenAI_Final_Project_City_of_Boston_RAG
Final project for NU CS6180 Foundations of Generative AI on RAG for City of Boston Public Notices

## Team Members
Margaret Bertoni
Aidan Domondon
Mohammad Aasim Shaikh

## Project Description
We want to increase the accessibility of the City of Boston website to the general public while maintaining trust. The City of Boston's website currently has an AI Summary function, but it's cite-ability is very limited, particularly when asking about public notices (for official meetings).
The public notices are stored as webpages, but also often attach PDFs with additional information to them.
Our project is a multi-agentic AI chatbot that uses RAG for questions


## Tech Stack

PDF Extraction: Docling
RAG Documents: LangChain
Vector Store: ChromaDB


## RAG Schema
The page texts and PDFs are separately added as docs to the ChromaDB. 

**The metadata for each LangChain doc:**
notice_metadata = {
    "notice_id": ,
    "title":,
    "cancelled":
    "detail_url":,
    "posted_at":,
    "event_datetime":,
    "address_1":,
    "address_2":,
    "status":,
    "checked_at":,
}

**Metadata for PDFs**
"file_label": file name,
"file_hash":,
"source_type": "pdf"
**Metadata for Page Text:**
source_type": "page_text"
