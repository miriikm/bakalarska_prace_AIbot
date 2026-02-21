import re
from pathlib import Path

from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

_SCRIPT_DIR = Path(__file__).resolve().parent
FAISS_INDEX_STATIC = _SCRIPT_DIR / "faiss_index_static"
MANUAL_URL = "https://speciationgenomics.github.io"

FAISS_INDEX_STATIC.mkdir(exist_ok=True)


def _text_only(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<script[^>]*>[\s\S]*?</script>", " ", s, flags=re.I)
    s = re.sub(r"<style[^>]*>[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def main():
    print("Loading", MANUAL_URL, "...")
    loader = RecursiveUrlLoader(url=MANUAL_URL, max_depth=4, timeout=30)
    docs = loader.load()
    cleaned = []
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        text = _text_only(text)
        if len(text.strip()) < 50:
            continue
        d.page_content = text[:500000]
        cleaned.append(d)
    if not cleaned:
        raise SystemExit("No documents loaded from URL.")
    print("Building embeddings and FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(cleaned)
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(str(FAISS_INDEX_STATIC))
    print("Saved to", FAISS_INDEX_STATIC, "(" + str(len(chunks)) + " chunks)")


if __name__ == "__main__":
    main()
