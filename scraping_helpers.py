
import requests, json
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import os
import re
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from pathlib import Path
import shutil
import re
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Get the base URL for the Notices (And URL for archives)
#boston_landing = "https://www.boston.gov"
notice_landing = "https://www.boston.gov/public-notices"
archive_landing = "https://www.boston.gov/archived-public-notices"

# Ensure that the path for the PDFs exists
folder_name = "public-notice-pdfs"

# File path for logs
log_path = folder_name+"/"+"notice_logs"

# Function to write log
def log_notice(log_path:str, record):
    with open(log_path,"a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False)+"\n")


#Function to filter log to check file ids
def check_logs(log_path:str, notice_id:str):
    ''' Returns the last record for the given notice id if it exists, otherwise None'''
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in reversed(lines):
        record = json.loads(line)
        if str(record.get("notice_id")) == notice_id:
            return record
    return None
    
# Function to get text safely when scraping
def safe_get_text(container, tag, **kwargs):
    ''' Get text safely if there's an empty field'''
    found = container.find(tag, **kwargs)
    return found.get_text(strip=False) if found else ""

#Sanitize file names
def sanitize_filename(name):
    return re.sub(r'[\\/:*?"<>|]', '-', name)

#Hashing for Files and Text contents
def hash_sha256(data:bytes):
    return hashlib.sha256(data).hexdigest()

# Check to make download is PDF
def is_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"

#Get all the latest records
def load_latest_records(log_path):
    records = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records[str(r["notice_id"])] = r  # last line for an id wins
    return records

# Get the list of IDs from the folders
def get_ids_from_folders(folder_name, log_path):
    log_filename = os.path.basename(log_path)
    return [
        e for e in os.listdir(folder_name)
        if e != log_filename and os.path.isdir(os.path.join(folder_name, e))
    ]



# Functions to extract data from a Notice 
def extract_notice(notice_id:str, log_path:str):

    # Make sure file folder exists
    notice_folder = os.path.join(folder_name, notice_id)
    os.makedirs(notice_folder, exist_ok=True)
    # Get the link for the notice:
    notice_url = notice_landing+"/"+notice_id

    #Get the url contents
    response = requests.get(notice_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Start extracting 
    title=soup.title.string
    
    # Finding the Posted Date
    posted_label=soup.find("div",class_="dl-t", string=lambda t:t and "Posted" in t)
    posted_raw=posted_label.find_next_sibling("div",class_="dl-d").get_text(strip=True)
    posted_at = datetime.strptime(posted_raw, "%m/%d/%Y - %I:%M%p").replace(tzinfo=ZoneInfo("America/New_York")).isoformat()

    # Discussion Topics Text
    discussion_label = soup.find("h2",class_="header-border-bottom", string=lambda t:t and "Discussion Topics" in t)
    discussion_text = discussion_label.find_next_sibling("div",class_="body").get_text(strip=False)


    # Event details
    event_date_container = soup.find("div", class_="date-title")
    event_datetime = event_date_container.find("time")["datetime"]
    address_container = soup.find("div",class_="detail-item__body--secondary sb-d")
    address_line_1 = safe_get_text(address_container, "span", class_="address-line1")
    address_line_2 = safe_get_text(address_container, "span", class_="address-line2")

    #Look for public comment
    public_testimony = False
    testimony = soup.find("div",class_="n-li-a", string=lambda t:t and "The public can offer testimony" in t)
    if testimony:
        public_testimony = True

    # Look for cancellation
    cancelled = False
    cancellation = soup.find("span",class_="t--err t--s60pct", string=lambda t:t and "Canceled" in t)
    if cancellation:
        cancelled=True
    
    # PDFS
    files = []
    resources_label = soup.find("div", class_="sb-t", string=lambda t: t and "Resources" in t)
    if resources_label:
        resources_container = resources_label.find_parent("div", class_="detail-item__content")
        pdf_links = resources_container.select("div.link-wrapper.download-link a")

        files = [{"file_label": sanitize_filename(a.get_text(strip=True)), "file_url": a["href"]} for a in pdf_links]


        # Check if any files have been added
        ## Get the last record
        previous = check_logs(log_path, notice_id)

        # Check old urls (not applicable if new notice) 
        old_by_url = {f["file_url"]: f for f in previous.get("files", [])} if previous else {}
        new_urls = {f["file_url"] for f in files}
        needs_download = [
            f for f in files if f["file_url"] not in old_by_url or not old_by_url[f["file_url"]].get("download_success")
        ]
        removed_files = [f for url, f in old_by_url.items() if url not in new_urls]
        
        # Loop through the files:
        for file in files:
            if file in needs_download:
                response = requests.get(file["file_url"])
                if response.status_code == 200 and is_pdf(response.content):
                    file_path = os.path.join(folder_name, str(notice_id), file["file_label"])
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    file["download_success"] = True
                    file["file_hash"] = hash_sha256(response.content)
                else:
                    file["download_success"] = False
                    file["file_hash"] = None
            else:
                # already succeeded last time — stamp forward from the old record
                old = old_by_url[file["file_url"]]
                file["download_success"] = old["download_success"]
                file["file_hash"] = old["file_hash"]

        # Check if any files to delete
            if removed_files: 
                for file in removed_files:
                    removal_path = os.path.join(folder_name, str(notice_id), file["file_label"])
                    os.remove(removal_path)

    # Write to log TODO
    record = {
        "notice_id": notice_id,
        "title": title,
        "cancelled": cancelled,
        "public_testimony": public_testimony,
        "notice_url": notice_url,
        "posted_at": posted_at,
        "event_datetime": event_datetime,
        "address_1":address_line_1,
        "address_2":address_line_2,
        "page_text": discussion_text,
        "files": files,
        "status": "ok",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    log_notice(log_path, record)


#Get all the latest records
def load_latest_records(log_path):
    records = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                records[str(r["notice_id"])] = r  # last line for an id wins
    return records

# Get the list of IDs from the folders
def get_ids_from_folders(folder_name, log_path):
    log_filename = os.path.basename(log_path)
    return [
        e for e in os.listdir(folder_name)
        if e != log_filename and os.path.isdir(os.path.join(folder_name, e))
    ]


# Embeddings and ChromaDB
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "public_notices"
EXPORT_TYPE = ExportType.DOC_CHUNKS

# If want to tweak chunk size, do that here
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100
)

# MAKE THE CHROMADB
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma(
    collection_name = COLLECTION_NAME,
    embedding_function = embeddings,
    persist_directory=CHROMA_DB_PATH,
)

# Prepended to every chunk's text before embedding.
def chunk_header(meta):
    title = (meta.get("title") or "").replace(" | Boston.gov", "")
    return f"{title} | event {meta.get('event_datetime')}"


# Check if something is already embedded to Chroma
def already_embedded(vectorstore, notice_id, file_hash=None, text_hash=None):
    notice_dict = {"notice_id": notice_id}
    file_dict = {}
    text_dict = {}
    terms = {}
    if file_hash:
        file_dict["file_hash"] = file_hash
        terms["$and"] = [notice_dict, file_dict]
        
    if text_hash:
        text_dict["text_hash"] = text_hash
        terms["$and"] = [notice_dict, text_dict]
    existing = vectorstore.get(where=terms, limit=1)
    return len(existing["ids"]) > 0

# List of required fields for metadata to avoid bad records
REQUIRED_FIELDS = ["notice_id", "title", "cancelled", "public_testimony",
                    "notice_url", "posted_at", "event_datetime",
                    "address_1", "address_2", "status", "checked_at"]

