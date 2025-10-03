# Indian Constitution Chatbot (RAG)

This project contains a RAG (retrieval-augmented generation) pipeline built in a Jupyter notebook and a Streamlit frontend to interact with it.

Files added:
- `backend/rag_bot.py` — small wrapper to load the pickled FAISS index and run the RetrievalQA chain.
- `frontend/app.py` — Streamlit app to ask questions and display answers + sources.
- `requirements.txt` — Python dependencies to install.

Quick start
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows (Git Bash) use: source .venv/Scripts/activate
pip install -r requirements.txt
```

2. Ensure `vector_db.pkl` exists in the project root. If you haven't created it yet, open `ChatBOT_Indian_Constitution.ipynb` and run the cells that load the PDF, split documents, create embeddings, and pickle the vector DB.

3. Provide your Cohere API key either via the sidebar in the Streamlit app or by setting `COHERE_API_KEY` in your environment.

4. Run Streamlit:

```bash
streamlit run frontend/app.py
```

Notes
- The backend uses `langchain_cohere.ChatCohere` and the notebook used the `command-xlarge-nightly` model. Make sure your Cohere plan supports it.
- This is a minimal integration focused on reusing the notebook code with a simple frontend. You can extend `backend/rag_bot.py` to support caching, streaming responses, or different LLMs.
