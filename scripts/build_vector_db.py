import os
import pickle
import sys

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdf_path = os.path.join(repo_root, "constitution_of_india.pdf")
    out_pickle = os.path.join(repo_root, "vector_db.pkl")

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(2)

    print(f"Loading PDF from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    print("Creating embeddings (this may take a moment)...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS vector store...")
    vector_db = FAISS.from_documents(chunks, embedding_model)

    print(f"Saving vector DB to {out_pickle} ...")
    with open(out_pickle, "wb") as f:
        pickle.dump(vector_db, f)

    print("✅ vector_db.pkl created successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR during vector DB build:", e)
        raise
