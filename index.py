import os
import json
import math
import csv
import uuid
import faiss
import requests
import numpy as np
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# Llama Del Rey - Indexer
# Builds FAISS index from PDFs and CSVs under ./docs
# ============================================================

DOCS_DIR = Path("docs")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.jsonl"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE = 800       # characters
CHUNK_OVERLAP = 150    # characters


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        print(f"[Llama Del Rey] WARN: Failed to read PDF {path}: {e}")
        return ""


def read_csv_as_text(path: Path, max_rows: int = 20000) -> str:
    lines = []
    try:
        # try utf-8 first
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                if i >= max_rows:
                    break
                headers = r.fieldnames or row.keys()
                kv = [f"{k}: {row.get(k, '')}" for k in headers]
                lines.append(", ".join(kv))
    except UnicodeDecodeError:
        # fallback latin-1
        with open(path, newline="", encoding="latin-1") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                if i >= max_rows:
                    break
                headers = r.fieldnames or row.keys()
                kv = [f"{k}: {row.get(k, '')}" for k in headers]
                lines.append(", ".join(kv))
    except Exception as e:
        print(f"[Llama Del Rey] WARN: Failed to read CSV {path}: {e}")
    return "\n".join(lines)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call Ollama /api/embeddings for each text and return vectors."""
    vectors = []
    for t in texts:
        payload = {"model": EMBED_MODEL, "prompt": t}
        resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        vec = data.get("embedding") or (data.get("data", [{}])[0].get("embedding"))
        if not vec:
            raise RuntimeError(f"Unexpected embedding response: {data}")
        vectors.append(vec)
    return vectors


def l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    out = []
    for v in vecs:
        norm = math.sqrt(sum(x*x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def iter_docs(doc_dir: Path) -> List[Tuple[str, str, int, str, str]]:
    """
    Iterate over PDFs and CSVs, returning tuples:
      (doc_id, chunk_text, chunk_index, source_path, source_type)
    """
    items = []
    for path in sorted(doc_dir.glob("**/*")):
        if path.is_dir():
            continue
        ext = path.suffix.lower()
        if ext not in [".pdf", ".csv"]:
            continue

        if ext == ".pdf":
            full_text = read_pdf(path)
            source_type = "pdf"
        else:
            full_text = read_csv_as_text(path)
            source_type = "csv"

        chunks = chunk_text(full_text)
        for idx, ch in enumerate(chunks):
            if not ch.strip():
                continue
            doc_id = str(uuid.uuid4())
            items.append((doc_id, ch, idx, str(path), source_type))
    return items


def main():
    print("[Llama Del Rey] Scanning docs...")
    triples = iter_docs(DOCS_DIR)
    if not triples:
        print("[Llama Del Rey] No content found in docs/. Exiting.")
        return

    print(f"[Llama Del Rey] {len(triples)} chunks to embed...")
    texts = [t[1] for t in triples]
    vecs = embed_texts(texts)
    if not vecs:
        print("[Llama Del Rey] No embeddings generated. Exiting.")
        return

    # L2-normalize for cosine similarity via inner product
    vecs = l2_normalize(vecs)

    # Create FAISS index
    dim = len(vecs[0])
    index = faiss.IndexFlatIP(dim)

    # Add vectors to the index (order matches metadata writing below)
    mat = np.array(vecs, dtype="float32")
    index.add(mat)

    # Save FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[Llama Del Rey] Saved FAISS index -> {INDEX_PATH}")

    # Save metadata aligned to index rows
    with open(META_PATH, "w", encoding="utf-8") as f:
        for (doc_id, chunk_text_value, chunk_idx, source_path, source_type) in triples:
            meta = {
                "id": doc_id,
                "chunk_index": chunk_idx,
                "text": chunk_text_value,
                "source_path": source_path,
                "source_type": source_type,
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    print(f"[Llama Del Rey] Saved metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
