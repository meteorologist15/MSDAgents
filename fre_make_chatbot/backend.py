import os
import sys
import logging
from pathlib import Path
from typing import List, Callable, Optional, Dict

# Compatibility fix for older sqlite3
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import psycopg2
from sqlalchemy import make_url
from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings, 
    PromptTemplate,
    SimpleDirectoryReader,
    get_response_synthesizer,
    Document as LlamaDocument
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

try:
    from fre_database import FreDatabase
except ImportError:
    FreDatabase = None

# --- Configuration ---
OLLAMA_BASE_URL = "http://localhost:11434" 
MODEL_NAME = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

# Postgres Connection
DB_NAME = "gfdl_fremake_chatbot_pgdb"
DB_USER = "Kristopher.Rand"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"

# --- Enhanced Structured Prompting ---
GFDL_CHAT_PROMPT = (
    "You are a technical Assistant at the Geophysical Fluid Dynamics Laboratory (GFDL), an expert in the 'fre make' module of the FRE (Flexible Modeling Systems Runtime Environment) framework -- a workflow algorithm designed to optimize the compiling, running, and post-processing of GFDL-developed climate models.\n"
    "Your goal is to provide accurate, structured, and concise information to scientists running this workflow, specifically as it relates to the 'fre make' module.\n\n"
    "RESPONSE STRUCTURE:\n"
    "1. **Summary**: A 1-2 sentence overview of the answer.\n"
    "2. **Details**: Use bullet points for steps or parameters.\n"
    "3. **Example**: Provide a CLI command or config snippet ONLY if relevant.\n\n"
    "CONSTRAINTS:\n"
    "- Use Markdown headings (###) for sections.\n"
    "- Be verbose ONLY if the user asks for 'detailed explanation' or 'deep dive'. Otherwise, keep it functional.\n"
    "- NEVER mention internal Python script names (e.g., utils.py) unless asked about implementation.\n"
    "- If referencing chat history, ensure consistency with previous answers.\n"
)

def configure_settings():
    Settings.llm = Ollama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, request_timeout=180.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def get_vector_store():
    """Initializes the PGVectorStore."""
    return PGVectorStore.from_params(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        table_name="fremake_vectors",
        embed_dim=768
    )

def detect_module_target(directory_path: str) -> Optional[str]:
    """Dynamically detects the module target (e.g., 'make', 'pp') based on the path."""
    path_str = str(Path(directory_path).absolute()).lower()
    for target in ["make", "yaml", "app", "list", "pp", "run"]:
        if target in path_str:
            return target
    return None


def run_ingestion(directory_path: str, logger: Callable[[str], None] = print) -> int:
    configure_settings()
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    # Inject paths into sys.path to ensure 'import fre' and local modules resolve
    abs_dir = os.path.abspath(directory_path)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)

    parent_dir = os.path.dirname(abs_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    grandparent_dir = os.path.dirname(parent_dir)
    if grandparent_dir not in sys.path:
        sys.path.insert(0, grandparent_dir)

    all_llama_docs = []

    logger(f"Reading files from {directory_path}...")
    reader = SimpleDirectoryReader(
        input_dir=directory_path, 
        recursive=True, 
        required_exts=[".py", ".md", ".rst", ".txt"], 
        exclude=["**/__init__.py", "**/tests/*", "**/__pycache__/*"])

    file_docs = reader.load_data()
    
    for doc in file_docs:
        file_path = doc.metadata.get("file_path", "")
        file_name = os.path.basename(file_path)
        content = doc.get_content()

        if "extra_docs" in file_path or "README" in file_name.upper():
            doc.set_content(f"DOCUMENT TYPE: High-level Documentation\nFILE: {file_name}\nCONTENT:\n{content}")
            doc.metadata["type"] = "user_doc"
        elif file_path.endswith(".py"):
            doc.set_content(f"# SOURCE CODE FILE: {file_name}\n# IDENTITY: This is internal implmentation code.\n{content}")
            doc.metadata["type"] = "raw_source"

        all_llama_docs.append(doc)

    if FreDatabase:
        module_target = detect_module_target(directory_path)
        logger(f"Detected FRE module target: '{module_target or 'all'}'")
        logger("Initializing and running the Runtime Docstring Inspection database...")
        try:
            # Pass our dynamically detected module target to the modified class
            fre_db = FreDatabase(module_name=module_target)
            fre_db.summarize()
            doc_list, metadata_list, id_list = fre_db.to_chromadb()
            
            for doc_text, metadata_dict, unique_id in zip(doc_list, metadata_list, id_list):
                # We build a LlamaIndex document mapped to the exact doc_id to support clean updates/upserts
                llama_doc = LlamaDocument(
                    text=f"DOCUMENT TYPE: Structured Sphinx Documentation\nSOURCE MODULE: {metadata_dict.get('module')}\nCOMPONENT: {metadata_dict.get('name')}\nCONTENT:\n{doc_text}",
                    metadata={
                        "file_path": metadata_dict.get("module", ""),
                        "name": metadata_dict.get("name", ""),
                        "package": metadata_dict.get("package", "fre"),
                        "type": "parsed_docstring"
                    },
                    doc_id=unique_id
                )
                all_llama_docs.append(llama_doc)
            logger(f"Successfully processed and loaded {len(doc_list)} structured Sphinx elements.")
        except Exception as e:
            logger(f"⚠️ Warning: Custom DB parser skipped due to error: {e}. Standard file context preserved.")
    else:
        logger("⚠️ Note: 'fre_database.py' not found. Skipping Sphinx ingestion.")

    logger(f"Processing documentation streams into vector nodes...")
    logger("Initializing specialized splitters...")
    python_splitter = CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500)
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    nodes = []
    for doc in all_llama_docs:
        # If it's a pre-parsed docstring or markdown/text, we use standard text_splitter
        if doc.metadata.get("type") in ["parsed_docstring", "user_doc"]:
            nodes.extend(text_splitter.get_nodes_from_documents([doc]))
        # Use CodeSplitter for the raw implementation
        elif doc.metadata.get("type") == "raw_source":
            try:
                nodes.extend(python_splitter.get_nodes_from_documents([doc]))
            except Exception:
                nodes.extend(text_splitter.get_nodes_from_documents([doc]))
        else:
            nodes.extend(text_splitter.get_nodes_from_documents([doc]))

    logger(f"Syncing {len(nodes)} nodes to PostgreSQL database {DB_NAME}...")
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes, storage_context=storage_context, show_progress=True)
    return len(nodes)

def get_chat_engine():
    """Initializes a Chat Engine with Memory and Hybrid Search."""
    configure_settings()
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)
    
    retriever = QueryFusionRetriever(
        [index.as_retriever(similarity_top_k=10)],
        similarity_top_k=10,
        mode="reciprocal_rerank",
        use_async=False
    )
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
    
    return CondensePlusContextChatEngine.from_defaults(
        retriever=retriever,
        memory=memory,
        system_prompt=GFDL_CHAT_PROMPT,
        verbose=False
    )

def log_interaction(query: str, response: str, metadata: Dict = None):
    """Logs every interaction to Postgres for auditing and fine-tuning."""
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS interaction_logs ("
        "id SERIAL PRIMARY KEY, query TEXT, response TEXT, model_name TEXT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    cur.execute("INSERT INTO interaction_logs (query, response, model_name) VALUES (%s, %s, %s)", (query, response, MODEL_NAME))
    conn.commit()
    cur.close()
    conn.close()

def evaluate_response(query: str, response_obj) -> Dict:
    """Uses LLM to evaluate faithfulness and relevancy."""
    faith_eval = FaithfulnessEvaluator(llm=Settings.llm)
    rel_eval = RelevancyEvaluator(llm=Settings.llm)
    
    f_result = faith_eval.evaluate_response(response=response_obj)
    r_result = rel_eval.evaluate_response(query=query, response=response_obj)
    
    return {"faithfulness": f_result.passing, "relevancy": r_result.passing}

def save_feedback(query: str, response: str, score: int, feedback_text: str = ""):
    """Saves user feedback to a Postgres table."""
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS user_feedback (id SERIAL PRIMARY KEY, query TEXT, response TEXT, score INT, feedback TEXT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    cur.execute("INSERT INTO user_feedback (query, response, score, feedback) VALUES (%s, %s, %s, %s)", (query, response, score, feedback_text))
    conn.commit()
    cur.close()
    conn.close()
