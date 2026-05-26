# GFDL Assistant - FRE make RAG Chatbot

This tool is a RAG (Retrieval-Augmented Generation) chatbot designed to assist scientists and developers in navigating the GFDL fre make module documentation and source code. It runs locally using Ollama for LLM inference and embeddings.


## Prerequisites

- **Linux Environment**: Optimized for GFDL workstations.

- **Python 3.11+**: The setup script enforces a minimum version of 3.11.

- **Ollama Service**: Assumes Ollama is running as a service on localhost:11434.

- **Models**: Ensure these models are pulled in Ollama:

    - ```llama3.1:8b```

    - ```nomic-embed-text```


## Installation

1. Clone or copy this repository to your local machine. Ensure Python 3.11 is your default python.

2. Run the setup script to create the environment and install dependencies:

```
chmod +x setup_env.sh

./setup_env.sh
```

3. Source the environment:

```
source venv/bin/activate
```

## Usage

The application is managed through ```fre_make_chatbot/frontend.py```.

### Web Interface (Streamlit)

To launch the browser-based chat interface:

```
python fre_make_chatbot/frontend.py ui
```


### Data Ingestion

Before the chatbot can answer questions, you must index the fre make documentation and associated source code. You can do this via the sidebar in the Web UI or via the command line:

```
python fre_make_chatbot/frontend.py ingest /path/to/your/fre-cli/fre/make
```
OR
```python fre_make_chatbot/frontend.py ingest /home/Kristopher.Rand/fre-cli/fre/make```

### Command Line Interface (Interactive)

To chat directly in your terminal:

```
python fre_make_chatbot/frontend.py query
```

## Technical Details

- **Backend**: Langchain with Ollama.

- **Splitters**: Hybrid strategy using a Python code TextSplitter and a regular Text Splitter (Markdown/RST).

- **Storage**: Persistent PostgreSQL instance located in.
