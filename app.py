import os
import json
import math
import re
import faiss
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Optional: .env support
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# Llama Del Rey - API (/ask)
# Uses FAISS index + Ollama to answer with local RAG
# ============================================================

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.jsonl"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

TOP_K = 12
MAX_CONTEXT_CHARS = 4500  # allow a bit more context

# ---------- Load index + metadata ----------
print("[Llama Del Rey] Loading FAISS index and metadata...")
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise RuntimeError("[Llama Del Rey] Index or metadata not found. Run `python index.py` first.")

index = faiss.read_index(str(INDEX_PATH))

# metadata arrays aligned with index order (same order as written in index.py)
chunks_text: List[str] = []
sources: List[str] = []
source_types: List[str] = []

with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        chunks_text.append(obj.get("text", ""))
        sources.append(obj.get("source_path", ""))
        source_types.append(obj.get("source_type", ""))

print(f"[Llama Del Rey] Loaded {len(chunks_text)} chunks.")

# ---------- Schemas ----------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

class AskDebugResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

# ---------- Helpers ----------
def embed_query(text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}  # IMPORTANT: 'prompt' (not 'input')
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding") or (data.get("data", [{}])[0].get("embedding"))
    if not vec:
        raise RuntimeError(f"Unexpected embedding response: {data}")
    # L2-normalize for cosine via inner product
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]

def search_similar(qvec: List[float], top_k: int = TOP_K) -> List[int]:
    q = np.array([qvec], dtype="float32")
    scores, idxs = index.search(q, top_k)
    return [int(i) for i in idxs[0] if i != -1]

def keyword_rescue(question: str, limit: int = 3) -> List[int]:
    """
    Cheap keyword booster: bring top snippets containing the query terms.
    Helps when semantic retrieval misses exact phrasing.
    """
    q_tokens = [t for t in re.findall(r"\w+", question.lower()) if len(t) > 2]
    scores = []
    for i, txt in enumerate(chunks_text):
        t = txt.lower()
        hit = sum(t.count(w) for w in q_tokens)
        scores.append((hit, i))
    scores.sort(reverse=True)
    return [i for hit, i in scores if hit > 0][:limit]

def rerank_by_keywords(idxs: List[int], keywords: List[str]) -> List[int]:
    kw = [k.lower() for k in keywords]
    def score(i: int) -> int:
        t = chunks_text[i].lower()
        return sum(t.count(k) for k in kw)
    return sorted(idxs, key=score, reverse=True)

# --- NEW: simple language detection (PT vs EN) ---
def detect_language(text: str) -> str:
    """
    Very simple heuristic:
    - If it has Portuguese diacritics or common PT words -> 'pt'
    - Else if it has common EN words -> 'en'
    - Fallback: if it contains only ASCII, lean 'en'; otherwise 'pt'
    """
    s = (text or "").strip()
    s_lower = s.lower()

    # Portuguese markers: diacritics or frequent words
    if re.search(r"[áéíóúâêôãõç]", s_lower):
        return "pt"
    pt_words = {"benefícios", "colaboradores", "brasileiros", "direito", "tem", "aulas", "português", "portugues", "quais", "são", "sao"}
    if any(w in s_lower for w in pt_words):
        return "pt"

    # English markers: common words
    en_words = {"benefits", "employees", "foreign", "right", "do", "have", "lessons", "portuguese", "which", "are"}
    if any(w in s_lower for w in en_words):
        return "en"

    # ASCII-only bias to EN, else PT
    try:
        s.encode("ascii")
        return "en"
    except UnicodeEncodeError:
        return "pt"

def build_prompt(question: str, contexts: List[str]) -> str:
    # Build context block respecting size guard
    ctx_block = ""
    total = 0
    for i, c in enumerate(contexts, start=1):
        if total + len(c) > MAX_CONTEXT_CHARS:
            break
        ctx_block += f"[CTX {i}]\n{c}\n\n"
        total += len(c)

    lang = detect_language(question)

    # Task hint for benefit-style questions (both languages)
    needs_list = ("benef" in question.lower())  # matches 'benefícios' and 'benefits'
    if lang == "pt":
        system = (
            "Use exclusivamente o CONTEXTO a seguir para responder. "
            "Se a resposta não estiver no contexto, diga que não há informação suficiente. "
            "Responda em português (pt-BR). "
            "Não se apresente. Não faça perguntas de volta. Vá direto ao ponto."
        )
        task_hint = ""
        if needs_list:
            task_hint = (
                "Tarefa: liste exatamente os benefícios encontrados para o caso perguntado, "
                "mantendo os valores e moedas como estão no contexto."
            )
        user = (
            f"{task_hint}\n"
            f"Pergunta:\n{question}\n\n"
            f"CONTEXTO:\n{ctx_block}"
            "Resposta:"
        )
        return f"{system}\n\n{user}"
    else:
        system = (
            "Use ONLY the following CONTEXT to answer. "
            "If the answer is not present in the context, say there isn't enough information. "
            "Answer in English. "
            "Do not introduce yourself. Do not ask questions back. Be direct and concise."
        )
        task_hint = ""
        if needs_list:
            task_hint = (
                "Task: list exactly the benefits found for the asked case, "
                "preserving values and currencies as shown in the context."
            )
        user = (
            f"{task_hint}\n"
            f"Question:\n{question}\n\n"
            f"CONTEXT:\n{ctx_block}"
            "Answer:"
        )
        return f"{system}\n\n{user}"

def generate_answer(prompt: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096  # larger context window for safety
        },
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    if not r.ok:
        raise RuntimeError(f"Ollama generate failed {r.status_code}: {r.text}")
    data = r.json()
    return (data.get("response") or "").strip()

# ---------- FastAPI ----------
app = FastAPI(title="Llama Del Rey - Local RAG with Ollama + FAISS")

def prepare_context_indices(question: str, qvec: List[float]) -> List[int]:
    idxs = search_similar(qvec, TOP_K)

    # Keyword rescue first (exact phrasing often wins)
    extra = keyword_rescue(question, limit=3)
    merged = list(dict.fromkeys(extra + idxs))  # keep order, remove dups

    # Domain-specific reranking: give priority to benefits-related chunks (PT/EN)
    merged = rerank_by_keywords(
        merged,
        keywords=[
            # PT
            "benefícios para colaboradores brasileiros",
            "benefícios",
            "colaboradores brasileiros",
            "clt",
            "vale alimentação",
            "vale refeição",
            "gympass",
            "aulas de inglês",
            # EN
            "benefits",
            "brazilian employees",
            "meal allowance",
            "food allowance",
            "gympass",
            "english classes",
            "portuguese lessons",
        ],
    )
    return merged

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Field 'question' is required.")
    try:
        qvec = embed_query(q)
        idxs = prepare_context_indices(q, qvec)
        contexts = [chunks_text[i] for i in idxs if 0 <= i < len(chunks_text)]
        prompt = build_prompt(q, contexts)
        answer = generate_answer(prompt)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_debug", response_model=AskDebugResponse)
def ask_debug(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Field 'question' is required.")
    try:
        qvec = embed_query(q)
        idxs = prepare_context_indices(q, qvec)
        contexts = [chunks_text[i] for i in idxs if 0 <= i < len(chunks_text)]
        prompt = build_prompt(q, contexts)
        answer = generate_answer(prompt)
        srcs = []
        for i in idxs[:TOP_K]:
            if 0 <= i < len(chunks_text):
                srcs.append({
                    "source_path": sources[i],
                    "source_type": source_types[i],
                    "snippet": (chunks_text[i][:300] + ("..." if len(chunks_text[i]) > 300 else ""))
                })
        return AskDebugResponse(answer=answer, sources=srcs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
