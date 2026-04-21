#!/usr/bin/env python3

import argparse
import sys
import os
import subprocess
import requests
import backend as backend


def check_ollama():
    """Checks if the Ollama server is reachable."""
    try:
        response = requests.get(backend.OLLAMA_BASE_URL)
        return response.status_code == 200
    except:
        return False


def run_streamlit_app():
    """Launches the Streamlit interface."""
    import streamlit as st

    st.set_page_config(page_title="GFDL fre make Assistant", page_icon="❄️", layout="wide")
    st.title("GFDL Model Workflow Assistant")

    # Sidebar
    st.sidebar.header("System Status")
    if check_ollama():
        st.sidebar.success("✅ Ollama Online")
    else:
        st.sidebar.error("❌ Ollama Offline")

    st.sidebar.info(f"**Model:** {backend.MODEL_NAME}\n\n**Embed:** {backend.EMBED_MODEL}")
    st.sidebar.divider()
    
    st.sidebar.header("Data Management")
    data_dir = st.sidebar.text_input("Source Directory Path:", placeholder="/home/path/to/fre/make")
    if st.sidebar.button("Build/Update Index"):
        if data_dir:
            try:
                count = backend.run_ingestion(data_dir, logger=st.toast)
                st.cache_resource.clear()
                st.session_state["ingestion_done"] = count
                st.rerun()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
        else:
            st.sidebar.warning("Please provide a path.")

    if st.session_state.get("ingestion_done"):
        st.success(f"Indexed {st.session_state['ingestion_done']} nodes successfully!")
        del st.session_state["ingestion_done"]

    # Chat Interface
    engine = backend.get_query_engine()
    if engine:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about fre make usage or configuration..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing documentation..."):
                    response = engine.query(prompt)
                    st.markdown(response.response)
                    with st.expander("View Source Context"):
                        for node in response.source_nodes:
                            st.write(f"**File:** `{node.metadata.get('file_path')}`")
                            st.caption(f"Score: {node.score:.2f}")

            st.session_state.messages.append({"role": "assistant", "content": response.response})
    else:
        st.info("👈 Vector Database is empty. Use the sidebar to ingest code and documentation.")


def main():
    parser = argparse.ArgumentParser(description="GFDL Fre Make Chatbot CLI")
    subparsers = parser.add_subparsers(dest="command")

    # UI command
    subparsers.add_parser("ui", help="Launch the Streamlit web interface")

    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest data from terminal")
    ingest_p.add_argument("path", type=str, help="Path to code directory")

    # Query command
    query_p = subparsers.add_parser("query", help="Query from terminal (Interactive if no text)")
    query_p.add_argument("text", type=str, nargs="?", help="Question to ask")

    # If no arguments provided, show help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "ui":
        print("🌐 Launching Streamlit interface...")
        # Spawning streamlit as a subprocess
        # Using sys.exit(0) ensures the main argparse script doesn't fall through
        # Note: We use sys.executable to ensure the same python version is used
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        sys.exit(0)

    elif args.command == "ingest":
        print(f"Starting ingestion from: {args.path}")
        try:
            count = backend.run_ingestion(args.path)
            print(f"✅ Done. Indexed {count} nodes.")
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
        sys.exit(0)

    elif args.command == "query":
        engine = backend.get_query_engine()
        if not engine:
            print("❌ No index found. Run 'python llamaindex_frontend.py ingest <path>' first.")
            return

        if args.text:
            print(f"\nThinking...\n")
            print(f"Assistant > {engine.query(args.text).response}\n")
        else:
            print("\n" + "="*50)
            print("GFDL FRE-MAKE INTERACTIVE CHAT (TERMINAL)")
            print("Type 'exit' or 'quit' to close.")
            print("="*50)
            while True:
                try:
                    u = input("\nUser > ").strip()
                    if u.lower() in ['exit', 'quit']: 
                        print("Goodbye!")
                        break
                    if not u: continue
                    
                    print("Thinking...")
                    response = engine.query(u)
                    # Clear line and print response
                    print(" " * 20)
                    print(f"Assistant > {response.response}")
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
        sys.exit(0)

    else:
        parser.print_help()


if __name__ == "__main__":
    # Check if we are running within the Streamlit runtime
    # This is a much more robust check than searching sys.argv
    try:
        from streamlit.runtime import exists as st_exists
        if st_exists():
            run_streamlit_app()
            sys.exit(0)
    except ImportError:
        pass

    # Otherwise, handle CLI commands
    main()
