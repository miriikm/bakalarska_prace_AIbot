from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

_SCRIPT_DIR = Path(__file__).resolve().parent
STATIC_DOCS_DIR = _SCRIPT_DIR / "static_docs"
FAISS_INDEX_STATIC = _SCRIPT_DIR / "faiss_index_static"

STATIC_DOCS_DIR.mkdir(exist_ok=True)
FAISS_INDEX_STATIC.mkdir(exist_ok=True)


def main():
    all_docs = []
    for path in sorted(STATIC_DOCS_DIR.glob("*.txt")) + sorted(STATIC_DOCS_DIR.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if len(text.strip()) < 30:
                continue
            all_docs.append(Document(page_content=text[:120000], metadata={"source": str(path)}))
        except Exception:
            continue
    if not all_docs:
        raise SystemExit("V static_docs/ nejsou žádné .txt soubory. Nejprve spusť force_index.py nebo build_index.py.")
    print("Building embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(FAISS_INDEX_STATIC))
    print("Hotovo. Index uložen do", FAISS_INDEX_STATIC, "(" + str(len(chunks)) + " chunků).")


if __name__ == "__main__":
    main()
