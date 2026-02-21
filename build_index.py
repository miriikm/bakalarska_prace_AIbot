import re
import time
from pathlib import Path

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DOCS_DIR = _SCRIPT_DIR / "static_docs"
FAISS_INDEX_STATIC = _SCRIPT_DIR / "faiss_index_static"
MANUAL_URL = "https://speciationgenomics.github.io"

STATIC_DOCS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_STATIC.mkdir(exist_ok=True)


def _text_only(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", s, flags=re.I)
    s = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def fetch_and_save():
    for f in STATIC_DOCS_DIR.glob("*.txt"):
        f.unlink()
    loader = RecursiveUrlLoader(url=MANUAL_URL, max_depth=4, timeout=30)
    docs = loader.load()
    time.sleep(1)
    for i, doc in enumerate(docs):
        text = getattr(doc, "page_content", "") or ""
        text = _text_only(text)
        if len(text.strip()) < 50:
            continue
        path = STATIC_DOCS_DIR / f"manual_{i}.txt"
        path.write_text(text[:500000], encoding="utf-8")


def build_faiss():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    all_docs = []
    for path in list(STATIC_DOCS_DIR.glob("**/*.txt")) + list(STATIC_DOCS_DIR.glob("**/*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if len(text.strip()) < 30:
                continue
            all_docs.append(Document(page_content=text[:120000], metadata={"source": str(path)}))
        except Exception:
            continue
    if not all_docs:
        raise SystemExit("No documents in static_docs/. Run fetch first.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_STATIC))
    print(f"Saved FAISS index to {FAISS_INDEX_STATIC} ({len(chunks)} chunks)")


if __name__ == "__main__":
    print("Fetching", MANUAL_URL, "into static_docs/...")
    fetch_and_save()
    print("Building FAISS index in faiss_index_static/...")
    build_faiss()
    print("Done.")
