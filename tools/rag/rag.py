from dotenv import load_dotenv
load_dotenv()

import os
import pdfplumber
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
from langchain_classic.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# -------------------------
# 1 >> PDF Loading
# -------------------------
def load_pdf_text(path: str) -> str:
    """Extracts all selectable text from a PDF."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -------------------------
# 2 >> Text Splitter
# -------------------------
def split_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# -------------------------
# 3 >> Embeddings
# -------------------------
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# -------------------------
# 4 >> Process Multiple PDFs
# -------------------------
def process_pdfs(folder_path: str) -> list[dict]:
    """Extracts and splits text from all PDFs in a folder and adds source metadata."""
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = load_pdf_text(path)
            chunks = split_text(text)
            # store metadata with each chunk
            chunks_with_source = [{"text": chunk, "source": filename} for chunk in chunks]
            all_chunks.extend(chunks_with_source)
    return all_chunks

# -------------------------
# 5 >> Create or Load Vector Store
# -------------------------
def create_vector_store(chunks_with_source: list[dict], persist_dir: str = "./db") -> Chroma:
    """Creates and persists a Chroma vector store from text chunks."""
    texts = [c["text"] for c in chunks_with_source]
    metadatas = [{"source": c["source"]} for c in chunks_with_source]

    vector_db = Chroma.from_texts(
        texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=persist_dir
    )
    return vector_db

def load_vector_store(persist_dir: str = "./db") -> Chroma | None:
    """Load existing Chroma vector store if it exists."""
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    return None

# -------------------------
# 6 >> Run full pipeline
# -------------------------
folder_path = r"rag\resources"
# chunks_with_source = process_pdfs(folder_path)
# vector_store = create_vector_store(chunks_with_source)


vector_store = load_vector_store(persist_dir="./db")
if vector_store is None:
    chunks_with_source = process_pdfs(folder_path)
    vector_store = create_vector_store(chunks_with_source, persist_dir="./db")

# -------------------------
# 7 >> RAG Retrieval + QA
# -------------------------
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.5}
)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=1.5,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "map_reduce"
    retriever=retriever
)

# Example query
result = qa_chain.invoke({"query": "Who were members of atomquest team"})
print(result)
