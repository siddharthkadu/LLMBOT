import os
import pickle
from typing import List, Dict, Any

try:
    from langchain.chains import RetrievalQA
    from langchain_cohere import ChatCohere
except Exception as e:
    raise ImportError(
        "Required dependencies are not installed. Please run `pip install -r requirements.txt` in the project root. "
        "Original error: " + str(e)
    )


class RAGBot:
    """Small wrapper around the notebook RAG pipeline to reuse from Streamlit / notebooks.

    Loads a pickled FAISS vector store created by the notebook (`vector_db.pkl` by default),
    wires a Cohere LLM (ChatCohere) and exposes a simple `answer` method that returns
    the answer and source documents.
    """

    def __init__(self, vector_db_path: str = "vector_db.pkl", cohere_api_key: str | None = None, model: str = "command-xlarge-nightly", k: int = 3):
        # Optionally set the COHERE_API_KEY (if provided). If not provided, expect it to be in the env already.
        if cohere_api_key:
            os.environ["COHERE_API_KEY"] = cohere_api_key

        # Validate Cohere API key presence before trying to instantiate the client.
        if not os.environ.get("COHERE_API_KEY"):
            raise RuntimeError(
                "Cohere API key not found. Set the COHERE_API_KEY environment variable or pass cohere_api_key=<your_key> when creating the bot."
            )

        self.vector_db_path = vector_db_path
        self.model = model
        self.k = k

        self._load_vector_db()
        self._init_chain()

    def _load_vector_db(self) -> None:
        if not os.path.exists(self.vector_db_path):
            raise FileNotFoundError(
                f"Vector DB file not found at {self.vector_db_path}.\nRun the notebook cells that create and pickle the FAISS store (vector_db.pkl)."
            )

        with open(self.vector_db_path, "rb") as f:
            self.vector_db = pickle.load(f)

    def _init_chain(self) -> None:
        # Initialize Cohere LLM and RetrievalQA similar to the notebook
        try:
            self.llm = ChatCohere(model=self.model)
        except Exception as e:
            # Surface a clearer error message for missing/invalid keys
            raise RuntimeError(
                f"Failed to initialize Cohere LLM. Make sure COHERE_API_KEY is set and valid. Original error: {e}"
            )
        self.retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
        )

        # Also keep a raw Cohere client for simple chat-only replies (non-RAG)
        try:
            import cohere as _cohere
            self._cohere_client = _cohere.Client(os.environ.get("COHERE_API_KEY"))
        except Exception:
            # If the cohere client isn't available, leave None; RAG still works via langchain wrapper
            self._cohere_client = None

    def set_k(self, k: int) -> None:
        """Change number of retrieved docs used for answers and refresh retriever."""
        self.k = k
        self.retriever = self.vector_db.as_retriever(search_type="similarity", search_kwargs={"k": self.k})
        # Recreate chain to pick up retriever change
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
        )

    def answer(self, query: str) -> Dict[str, Any]:
        """Run query through the retrieval-augmented chain.

        Returns a dict: {"answer": str, "sources": List[dict]}
        Each source dict contains at least 'page_content' and any metadata present.
        """
        # Decide whether to use RAG or just the LLM.
        if not self._is_constitution_query(query):
            # Use simple LLM chat reply (no RAG) for greetings, small talk, etc.
            if self._cohere_client is not None:
                try:
                    resp = self._cohere_client.chat(message=query, model=self.model)
                    # cohere NonStreamedChatResponse holds text in resp
                    text = getattr(resp, "text", None)
                    if not text:
                        # some versions return object differently; try str()
                        text = str(resp)
                    return {"answer": text, "sources": []}
                except Exception:
                    # Fallback to RAG chain if chat fails
                    pass

        # Default: use the retrieval-augmented QA chain
        result = self.qa_chain({"query": query})
        answer_text = result.get("result") or result.get("answer") or ""
        source_docs = result.get("source_documents", [])

        sources = []
        for d in source_docs:
            sources.append({
                "page_content": getattr(d, "page_content", str(d))[:4000],
                "metadata": getattr(d, "metadata", {}),
            })

        return {"answer": answer_text, "sources": sources}

    def _is_constitution_query(self, query: str) -> bool:
        """Very small heuristic classifier: return True when the query looks like it's about the Constitution.

        This avoids calling the retriever/LLM for generic chit-chat. It's intentionally simple and
        can be improved later with a classifier or embedding-similarity check.
        """
        q = (query or "").lower().strip()

        # common constitution-related keywords -> definitely constitution-related
        constitution_keywords = [
            "constitution",
            "article",
            "section",
            "amendment",
            "fundamental right",
            "fundamental rights",
            "directive principle",
            "clause",
            "part ",
            "what is article",
            "which article",
            "right to",
        ]
        for k in constitution_keywords:
            if k in q:
                return True

        # explicit chit-chat / non-constitution triggers -> treat as non-constitution
        non_constitution_triggers = [
            "joke",
            "tell me a joke",
            "jokes",
            "riddle",
            "weather",
            "time",
            "date",
            "news",
            "who are you",
            "what is your name",
            "your name",
            "thanks",
            "thank you",
            "bye",
            "goodbye",
        ]
        for t in non_constitution_triggers:
            if t in q:
                return False

        # common short greetings -> not constitution
        greetings = ["hello", "hi", "hey", "good morning", "good evening", "how are you"]
        if any(q == g or q.startswith(g + " ") for g in greetings):
            return False

        # If the query explicitly mentions India + constitution/article phrasing
        if "india" in q and ("article" in q or "constitution" in q):
            return True

        # If it's a very short query, assume non-constitution unless it starts with a question word
        words = q.split()
        question_words = {"what", "who", "which", "how", "when", "where"}
        if len(words) < 4 and (not words or words[0] not in question_words):
            return False

        # default conservative: assume constitution-related (safe fallback)
        return True


# Module-level default bot (lazy-initialized)
_DEFAULT_BOT: RAGBot | None = None


def get_default_bot(vector_db_path: str = "vector_db.pkl", cohere_api_key: str | None = None, model: str = "command-xlarge-nightly", k: int = 3) -> RAGBot:
    global _DEFAULT_BOT
    if _DEFAULT_BOT is None:
        _DEFAULT_BOT = RAGBot(vector_db_path=vector_db_path, cohere_api_key=cohere_api_key, model=model, k=k)
    return _DEFAULT_BOT
