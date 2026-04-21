import os
import sys
from pathlib import Path
from typing import List, Callable, Optional

# Compatibility fix for older sqlite3 versions (common in HPC/Workstation environments)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings, 
    PromptTemplate,
    SimpleDirectoryReader
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434" 
MODEL_NAME = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = "./fremake_chroma_db"
COLLECTION_NAME = "fremake_collection"

# --- Custom Prompting for Scientist-Focused Context ---
GFDL_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "You are a technical expert for the GFDL (Geophysical Fluid Dynamics Laboratory) 'fre make' module and the wider 'FRE' (Flexible Modeling System (FMS) Runtime Environment) framework. Your primary audience consists of scientists.\n\n"
    "GUIDELINES FOR YOUR RESPONSE:\n"
    "1. PRIORITIZE information from files labeled with 'High-level Documentation', particularly for general overview questions.\n"
    "2. FOCUS on command-line usage (CLI), workflows, and user-facing configuration.\n"
    "3. DO NOT provide Python code snippets or internal implementation details unless requested.\n"
    "4. DO NOT mention internal script filenames (e.g., 'fremake.py', 'checkout.py') or directory paths.\n"
    "5. REFER to features by their functional names or CLI commands only.\n"
    "6. Use a professional, technical, yet accessible tone for a scientific research environment.\n\n"
    "Query: {query_str}\n"
    "Answer: "
)
GFDL_PROMPT = PromptTemplate(GFDL_PROMPT_TMPL)


def configure_settings():
    """Initializes global LlamaIndex settings for LLM and Embeddings."""
    Settings.llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, request_timeout=180.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)


def run_ingestion(directory_path: str, logger: Callable[[str], None] = print) -> int:
    """Parses files and builds the vector index."""
    configure_settings()
    
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    logger("Initializing specialized splitters...")
    python_splitter = CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    logger(f"Reading files from {directory_path}...")
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        recursive=True,
        required_exts=[".py", ".md", ".rst", ".txt"],
        exclude=["**/__init__.py", "**/tests/*"]
    )
    documents = reader.load_data()
    
    # --- Metadata Injection Step ---
    # We modify the text of the documents BEFORE they are split and embedded.
    # This ensures the 'Identity' of the file is part of the vector representation.
    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        file_name = os.path.basename(file_path)
        content = doc.get_content()
        
        if "extra_docs" in file_path or "README" in file_name.upper():
            # Force high-relevance tokens for documentation
            new_text = f"DOCUMENT TYPE: High-level Documentation and Overview\nFILE NAME: {file_name}\nCONTENT:\n{content}"
            doc.set_content(new_text)
        elif file_path.endswith(".py"):
            new_text = f"# SOURCE CODE FILE: {file_name}\n# IDENTITY: This is internal implementation code.\n{content}"
            doc.set_content(new_text)

    logger(f"Processing {len(documents)} source files into nodes...")
    nodes = []
    for doc in documents:
        file_ext = Path(doc.metadata.get("file_path", "")).suffix
        if file_ext == ".py":
            try:
                nodes.extend(python_splitter.get_nodes_from_documents([doc]))
            except Exception as e:
                logger(f"⚠️ Warning: Failed to parse {doc.metadata.get('file_path')} as Python. Falling back to text splitter.")
                nodes.extend(text_splitter.get_nodes_from_documents([doc]))
        else:
            nodes.extend(text_splitter.get_nodes_from_documents([doc]))

    logger(f"Syncing {len(nodes)} nodes to ChromaDB at {CHROMA_PATH}...")
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    return len(nodes)


def get_query_engine():
    """Initializes and returns the query engine from existing storage."""
    configure_settings()
    
    if not os.path.exists(CHROMA_PATH):
        return None

    try:
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        chroma_collection = db.get_collection(COLLECTION_NAME)
        if chroma_collection.count() == 0:
            return None
            
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        return index.as_query_engine(
            similarity_top_k=12, 
            text_qa_template=GFDL_PROMPT
        )
    except Exception:
        return None
