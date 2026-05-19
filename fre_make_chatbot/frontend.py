#!/usr/bin/env python3

import argparse
import sys
import subprocess
import requests
import threading
import itertools
import time

import asyncio
import nest_asyncio

# Force standard asyncio event loop policy before patching,
# as nest_asyncio is incompatible with Streamlit's default uvloop.
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()

import backend as backend

def check_ollama():
    """Checks if the Ollama server is reachable."""
    try:
        response = requests.get(backend.OLLAMA_BASE_URL)
        return response.status_code == 200
    except:
        return False

# --- CLI Spinner Helper --- #
class CLISpinner:
    def __init__(self, message="Thinking..."):
        self.message = message
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._animate)

    def _animate(self):
        for char in itertools.cycle(['|', '/', '-', '\\']):
            if self.stop_event.is_set():
                break
            sys.stdout.write(f'\r{self.message} {char} ')
            sys.stdout.flush()
            time.sleep(0.1)

        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join()

def run_streamlit_app():
    """Launches the enhanced Streamlit interface with Chat Memory and Evaluation."""
    import streamlit as st

    st.set_page_config(page_title="GFDL Assistant Pro", page_icon="❄️", layout="wide")
    st.title("GFDL Model Workflow Assistant (Streaming Mode)")

    st.sidebar.header("System Status")
    if check_ollama():
        st.sidebar.success("✅ Ollama Online")
    else:
        st.sidebar.error("❌ Ollama Offline")

    st.sidebar.info(f"**Model:** {backend.MODEL_NAME}\n\n**Embed:** {backend.EMBED_MODEL}")
    st.sidebar.divider()
    
    st.sidebar.header("Data Management")
    data_dir = st.sidebar.text_input("Source Directory Path:", placeholder="/home/path/to/fre/make")
    if st.sidebar.button("Build/Update PG Index"):
        if data_dir:
            try:
                count = backend.run_ingestion(data_dir, logger=st.toast)
                st.cache_resource.clear()
                st.sidebar.success(f"Indexed {count} nodes into Postgres (pgvector)!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
        else:
            st.sidebar.warning("Please provide a path.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    @st.cache_resource
    def get_engine():
        return backend.get_chat_engine()

    engine = get_engine()

    # --- Restore Historical Chat Messages ---
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                # 1. Restore Source Context Display for previous messages
                sources = msg.get("sources")
                if sources:
                    with st.expander("View Source Context"):
                        for source in sources:
                            st.write(f"**Source:** `{source['file']}` (Relevance: {source['score']:.2f})")

                # 2. Display stored evaluation scores
                eval_data = msg.get("eval")
                if eval_data:
                    f_status = "✅ Pass" if eval_data.get("faithfulness") else "❌ Fail"
                    r_status = "✅ Pass" if eval_data.get("relevancy") else "❌ Fail"
                    st.caption(f"**Faithfulness:** {f_status} | **Relevancy:** {r_status}")

                # 3. Firefox-Friendly Layout for Feedback Buttons (Integer Ratios)
                btn_col1, btn_col2, _ = st.columns([1, 1, 8])
                with btn_col1:
                    if st.button("👍", key=f"up_{i}", help="Correct or helpful"):
                        query_text = st.session_state.messages[i-1]["content"] if i > 0 else "Unknown"
                        backend.save_feedback(query_text, msg["content"], 1)
                        st.toast("Liked!")
                with btn_col2:
                    if st.button("👎", key=f"down_{i}", help="Incorrect or unhelpful"):
                        query_text = st.session_state.messages[i-1]["content"] if i > 0 else "Unknown"
                        backend.save_feedback(query_text, msg["content"], 0)
                        st.toast("Disliked!")

    # --- Handle New User Input ---
    if engine:
        if prompt := st.chat_input("Ask about fre make usage or configuration..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing context and generating response..."):
                    response_obj = engine.stream_chat(prompt)

                # Stream Output
                ans_text = st.write_stream(response_obj.response_gen)

                # Extract and Display Source Nodes dynamically
                source_data = []
                if hasattr(response_obj, 'source_nodes'):
                    for node in response_obj.source_nodes:
                        src = node.metadata.get('file_path', 'Internal Source')
                        score = node.score if node.score else 0.0
                        source_data.append({"file": src, "score": score})

                with st.expander("View Source Context"):
                    for source in source_data:
                        st.write(f"**Source:** `{source['file']}` (Relevance: {source['score']:.2f})")
                    
                # Evaluate and Log
                eval_results = backend.evaluate_response(prompt, response_obj)
                backend.log_interaction(prompt, ans_text)
                    
                # Render Eval Immediately
                f_status = "✅ Pass" if eval_results.get("faithfulness") else "❌ Fail"
                r_status = "✅ Pass" if eval_results.get("relevancy") else "❌ Fail"
                st.caption(f"**Faithfulness:** {f_status} | **Relevancy:** {r_status}")                   

                # Store EVERYTHING in Session State so it doesn't collapse/disappear on rerun
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ans_text,
                    "sources": source_data,
                    "eval": eval_results
                })
                
                # Rerun to render the feedback buttons for the newly generated message
                st.rerun()
    else:
        st.info("👈 Please use the sidebar to ingest your code into the PostgreSQL database.")


def main():
    parser = argparse.ArgumentParser(description="GFDL FRE make assistant")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ui", help="Launch the Streamlit web interface")
    ingest_p = subparsers.add_parser("ingest", help="Ingest data from terminal")
    ingest_p.add_argument("path", type=str, help="Path to code directory")
    query_p = subparsers.add_parser("query", help="Interactive terminal chat")
    query_p.add_argument("text", type=str, nargs="?", help="Question to ask")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "ui":
        print("🌐 Launching Streamlit interface...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        sys.exit(0)

    elif args.command == "ingest":
        print(f"Starting ingestion into pgvector from: {args.path}")
        try:
            count = backend.run_ingestion(args.path)
            print(f"✅ Done! Indexed {count} nodes.")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")

    elif args.command == "query":
        engine = backend.get_chat_engine()
        if not engine:
            print("❌ No PG index found! Run 'python frontend.py ingest <path>' first.")
            return

        if args.text:
            print(f"\nThinking...\n")
            print("Assistant > ", end="", flush=True)
            response_obj = engine.stream_chat(args.text)

            full_response = ""
            for token in response_obj.response_gen:
                print(token, end="", flush=True)
                full_response += token
            print()
        else:
            print("\n" + "="*50)
            print("GFDL INTERACTIVE TERMINAL CHAT")
            print("Type 'exit' or 'quit' to close.")
            print("="*50)

            while True:
                try:
                    u = input("\nUser > ").strip()
                    if u.lower() in ['exit', 'quit']: 
                        print("Goodbye!")
                        break
                    if not u: continue

                    with CLISpinner("Thinking..."):
                        response_obj = engine.stream_chat(u)

                    print("Assistant > ", end="", flush=True)
                    full_response = ""
                    for token in response_obj.response_gen:
                        print(token, end="", flush=True)
                        full_response += token
                    print()

                    backend.log_interaction(u, full_response)
                except KeyboardInterrupt:
                    break
        sys.exit(0)
    else:
        parser.print_help()

if __name__ == "__main__":
    # If Streamlit is running this script, we execute the UI and do absolutely nothing else.
    # This guarantees the argparse help screen will never trigger in the terminal during a browser session.
    try:
        from streamlit.runtime import exists as st_exists
        in_streamlit = st_exists()
    except ImportError:
        in_streamlit = False

    if in_streamlit:
        run_streamlit_app()
    else:
        main()
