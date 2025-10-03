import os
import streamlit as st
from backend.rag_bot import get_default_bot

import cohere



st.set_page_config(page_title="Indian Constitution Chatbot", layout="wide")

with st.sidebar:
    st.title("Settings")
    vector_path = st.text_input("Vector DB pickle path", value="vector_db.pkl")
    cohere_key = st.text_input("Cohere API Key (or leave blank to use env)", type="password")
    k = st.slider("Number of retrieved documents (k)", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.write("You can set COHERE_API_KEY in your environment instead of pasting here.\nExample (PowerShell):")
    st.code("$env:COHERE_API_KEY=\"your_key_here\"")
    st.write("Example (Git Bash):")
    st.code("export COHERE_API_KEY=your_key_here")
    st.markdown("---")
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if st.session_state.connected:
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.session_state.bot = None
            st.session_state.messages = []
    else:
        if st.button("Validate Key"):
            # Validate key using Cohere client before connecting
            # Normalize pasted key: trim whitespace and surrounding quotes
            key_to_test = cohere_key.strip() if cohere_key else None
            if key_to_test:
                if (key_to_test.startswith('"') and key_to_test.endswith('"')) or (key_to_test.startswith("'") and key_to_test.endswith("'")):
                    key_to_test = key_to_test[1:-1].strip()
            # safe access to st.secrets which can raise if no secrets file exists
            def _get_secret_safe(name: str):
                try:
                    return st.secrets.get(name)
                except Exception:
                    return None

            if not key_to_test and not _get_secret_safe("COHERE_API_KEY") and not st.query_params.get("COHERE_API_KEY"):
                # fall back to env var check
                if not os.environ.get("COHERE_API_KEY"):
                    st.error("No key provided. Paste your Cohere key in the sidebar or set COHERE_API_KEY in the environment.")
                    st.stop()
                else:
                    key_to_test = os.environ.get("COHERE_API_KEY")

            try:
                client = cohere.Client(key_to_test)
                # Use the Chat API for validation; fall back to generate if chat is missing
                if hasattr(client, 'chat'):
                    # simple chat call using the SDK's `message` parameter
                    client.chat(message="hi", model="command-xlarge-nightly")
                else:
                    client.generate(model="command-xlarge-nightly", prompt="hi", max_tokens=1)
            except Exception as ce:
                # Provide actionable guidance for 401 invalid api token
                msg = str(ce)
                if '401' in msg or 'invalid api token' in msg.lower():
                    st.error("Cohere validation failed: invalid API token (401).\n"+
                             "Possible fixes: paste the exact key (no quotes), regenerate a key in the Cohere dashboard, or set COHERE_API_KEY in the same shell used to start Streamlit.")
                else:
                    st.error(f"Cohere validation failed: {ce}")
            else:
                # validation passed; now try to initialize the bot
                try:
                    st.session_state.bot = get_default_bot(vector_db_path=vector_path, cohere_api_key=key_to_test if key_to_test else None, k=k)
                    st.session_state.messages = []
                    st.session_state.connected = True
                    st.success("Connected and validated Cohere key")
                except Exception as e:
                    st.error(f"Failed to initialize bot after validation: {e}")

st.header("Indian Constitution Chatbot (RAG)")
st.write("Ask questions about the Constitution of India. Uses FAISS + Cohere via LangChain.")

if not st.session_state.get("connected"):
    st.info("Not connected. Enter a Cohere API key in the sidebar and click Connect, or set COHERE_API_KEY in your environment and click Connect.")
    st.stop()

# update k if changed
if st.session_state.bot.k != k:
    st.session_state.bot.set_k(k)


def add_user_message(text: str):
    st.session_state.messages.append({"role": "user", "text": text})


def add_bot_message(text: str, sources=None):
    st.session_state.messages.append({"role": "bot", "text": text, "sources": sources or []})


col1, col2 = st.columns([3, 1])

with col1:
    # Chat area
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Bot:** {msg['text']}")
            if msg.get("sources"):
                for i, s in enumerate(msg["sources"], 1):
                    with st.expander(f"Source {i} - preview"):
                        if s.get("metadata"):
                            st.write(s["metadata"])
                        st.write(s.get("page_content", "")[:3000])

with col2:
    st.markdown("---")
    st.write("Quick actions")
    if st.button("Clear chat"):
        st.session_state.messages = []

user_input = st.text_input("", placeholder="Ask a question about the Indian Constitution...")
send = st.button("Send")

if send and user_input.strip():
    add_user_message(user_input)
    with st.spinner("Thinking..."):
        try:
            resp = st.session_state.bot.answer(user_input)
        except Exception as e:
            st.error(f"Error: {e}")
        else:
            add_bot_message(resp.get("answer"), resp.get("sources"))

