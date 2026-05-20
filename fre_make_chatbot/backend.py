import os
import sys
import logging
from pathlib import Path
from typing import List, Callable, Optional, Dict

import psycopg2

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- Dynamic Import for Colleague's Parser ---
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

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Prompting ---
GFDL_SYSTEM_PROMPT = (
    "You are a technical Assistant at the Geophysical Fluid Dynamics Laboratory (GFDL), an expert in the 'fre make' module of the FRE (Flexible Modeling Systems Runtime Environment) framework -- a workflow algorithm designed to optimize the compiling, running, and post-processing of GFDL-developed climate models.\n"
    "Your goal is to provide accurate, structured, and concise information to scientists running this workflow, specifically as it relates to the 'fre make' module.\n\n"
    "RESPONSE STRUCTURE:\n"
    "1. **Summary**: A 1-2 sentence overview of the answer.\n"
    "2. **Details**: Use bullet points for steps or parameters.\n"
    "3. **Example**: Provide a CLI command or config snippet ONLY if relevant.\n\n"
    "CONSTRAINTS:\n"
    "- Use Markdown headings (###) for sections.\n"
    "- Prioritize information from files labeled 'Structured Sphinx Documentation'.\n"
    "- Be verbose ONLY if the user asks for 'detailed explanation' or 'deep dive'. Otherwise, keep it functional.\n"
    "- NEVER mention internal Python script names (e.g., utils.py) unless asked about implementation.\n"
    "- If referencing chat history, ensure consistency with previous answers.\n\n"
    "CONTEXT:\n"
    "{context}"
)

def get_vector_store():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="fremake_vectors",
        use_uuid=True
    )

def detect_module_target(directory_path: str) -> Optional[str]:
    path_str = str(Path(directory_path).absolute()).lower()
    for target in ["make", "yaml", "app", "list", "pp", "run"]:
        if target in path_str:
            return target
    return None

def run_ingestion(directory_path: str, logger: Callable[[str], None] = print) -> int:
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    # Inject paths into sys.path to ensure local imports resolve
    abs_dir = os.path.abspath(directory_path)
    if abs_dir not in sys.path:
        sys.path.insert(0, abs_dir)
    parent_dir = os.path.dirname(abs_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    all_lc_docs = []
    logger("Scanning directory for files...")
    
    # Standard File Ingestion using os.walk
    valid_exts = [".py", ".md", ".rst", ".txt"]
    for root, _, files in os.walk(directory_path):
        if "__init__.py" in root or "tests" in root:
            continue
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in valid_exts:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    if "extra_docs" in file_path or "README" in file.upper():
                        page_content = f"DOCUMENT TYPE: High-level Documentation\nFILE: {file}\nCONTENT:\n{content}"
                        doc_type = "user_doc"
                    elif file.endswith(".py"):
                        page_content = f"# SOURCE CODE IMPLEMENTATION FILE\n# FILE: {file}\n{content}"
                        doc_type = "raw_source"
                    else:
                        page_content = content
                        doc_type = "general"
                        
                    all_lc_docs.append(Document(page_content=page_content, metadata={"file_path": file_path, "type": doc_type}))
                except Exception as e:
                    logger(f"Skipped {file_path}: {e}")

    # Sphinx Docstring Ingestion
    if FreDatabase:
        module_target = detect_module_target(directory_path)
        logger(f"Detected FRE module target: '{module_target or 'all'}'")
        try:
            fre_db = FreDatabase(module_name=module_target)
            fre_db.summarize()
            doc_list, metadata_list, id_list = fre_db.to_chromadb()
            
            for doc_text, metadata_dict, unique_id in zip(doc_list, metadata_list, id_list):
                all_lc_docs.append(Document(
                    page_content=f"DOCUMENT TYPE: Structured Sphinx Documentation\nSOURCE MODULE: {metadata_dict.get('module')}\nCOMPONENT: {metadata_dict.get('name')}\nCONTENT:\n{doc_text}",
                    metadata={
                        "file_path": metadata_dict.get("module", ""),
                        "name": metadata_dict.get("name", ""),
                        "package": metadata_dict.get("package", "fre"),
                        "type": "parsed_docstring",
                        "doc_id": unique_id
                    }
                ))
            logger(f"Successfully loaded {len(doc_list)} structured Sphinx elements.")
        except Exception as e:
            logger(f"⚠️ Custom DB parser skipped due to error: {e}")

    # Split Texts
    logger("Parsing documentation into vector nodes...")
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1500, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)

    nodes = []
    for doc in all_lc_docs:
        if doc.metadata.get("type") in ["parsed_docstring", "user_doc", "general"]:
            nodes.extend(text_splitter.split_documents([doc]))
        elif doc.metadata.get("type") == "raw_source":
            try:
                nodes.extend(python_splitter.split_documents([doc]))
            except:
                nodes.extend(text_splitter.split_documents([doc]))

    # Store in PGVector
    logger(f"Syncing {len(nodes)} vector nodes to PostgreSQL database: '{DB_NAME}'...")
    vector_store = get_vector_store()
    
    import uuid
    ids = [n.metadata.get("doc_id") if n.metadata.get("doc_id") else str(uuid.uuid4()) for n in nodes]
    vector_store.add_documents(nodes, ids=ids)
    
    return len(nodes)

class DocumentWrapper:
    """Wrapper to map LangChain Document attributes to the format expected by the frontend."""
    def __init__(self, doc):
        self.metadata = doc.metadata
        self.page_content = doc.page_content
        self.score = doc.metadata.get('score', 0.0)

class LCELChatEngineWrapper:
    """Wraps LangChain Retrieval chain to perfectly mimic LlamaIndex streaming behavior."""
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.chat_history = []
        self.source_nodes = []
        self._full_response = ""

    def stream_chat(self, query: str):
        self.source_nodes = []
        self._full_response = ""
        
        def generator():
            for chunk in self.rag_chain.stream({"input": query, "chat_history": self.chat_history}):
                if "context" in chunk:
                    self.source_nodes = [DocumentWrapper(d) for d in chunk["context"]]
                if "answer" in chunk:
                    self._full_response += chunk["answer"]
                    yield chunk["answer"]
            
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=self._full_response))
            
        self.response_gen = generator()
        return self

def get_chat_engine():
    """Initializes LCEL RAG Chain with Memory and Contextual compression."""
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

        # 1. Condense Question Prompt (History-Aware)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # 2. Answer Generator Prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", GFDL_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return LCELChatEngineWrapper(rag_chain)
    except Exception as e:
        print(f"Error initializing LangChain engine: {e}")
        return None

def log_interaction(query: str, response: str, metadata: Dict = None):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS interaction_logs (id SERIAL PRIMARY KEY, query TEXT, response TEXT, model_name TEXT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cur.execute("INSERT INTO interaction_logs (query, response, model_name) VALUES (%s, %s, %s)", (query, response, MODEL_NAME))
        conn.commit(); cur.close(); conn.close()
    except:
        pass

def evaluate_response(query: str, response_obj) -> Dict:
    """Custom LangChain implementation of Faithfulness and Relevancy evaluations."""
    try:
        llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.0)
        context_str = "\n\n".join([doc.page_content for doc in response_obj.source_nodes])
        response_text = response_obj._full_response
        
        f_prompt = f"Context: {context_str}\n\nResponse: {response_text}\n\nIs the Response fully supported by the Context? Answer strictly 'PASS' if yes, or 'FAIL' if no or if it hallucinates."
        f_res = llm.invoke(f_prompt).content
        
        r_prompt = f"Query: {query}\n\nResponse: {response_text}\n\nDoes the Response adequately answer the Query? Answer strictly 'PASS' if yes, or 'FAIL' if no."
        r_res = llm.invoke(r_prompt).content
        
        return {
            "faithfulness": "PASS" in f_res.upper(),
            "relevancy": "PASS" in r_res.upper()
        }
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"faithfulness": True, "relevancy": True}

def save_feedback(query: str, response: str, score: int, feedback_text: str = ""):
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS user_feedback (id SERIAL PRIMARY KEY, query TEXT, response TEXT, score INT, feedback TEXT, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cur.execute("INSERT INTO user_feedback (query, response, score, feedback) VALUES (%s, %s, %s, %s)", (query, response, score, feedback_text))
        conn.commit(); cur.close(); conn.close()
    except:
        pass
