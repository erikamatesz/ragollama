"""
Llama Del Rey - API (/ask) [PT-BR, multi-tema]
----------------------------------------------
Endpoint de Perguntas & Respostas usando RAG local (Ollama + FAISS).

Fluxo:
1) Carrega o índice FAISS e meta.jsonl (gerados pelo indexador).
2) Recebe uma pergunta em POST /ask.
3) Embedding da pergunta (Ollama /api/embeddings) + busca Top-K no FAISS.
4) Monta prompt em PT-BR com os melhores trechos (limite de contexto).
5) Gera a resposta com o LLM local (Ollama /api/generate).
6) /ask_debug devolve também as fontes/trechos usados.

Obs.: Esta versão é **multi-tema** e **PT-BR apenas**.
O rerank é dinâmico com base nas palavras/expressões da própria pergunta.
"""

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

# Suporte opcional a .env (não quebra se não existir)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# Configurações e caminhos
# ============================================================

DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.jsonl"

# Endpoints/modelos do Ollama (podem vir do .env)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Parâmetros do RAG (override via .env para facilitar a demo)
TOP_K = int(os.getenv("TOP_K", 12))  # quantos trechos similares buscar
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 4500))  # limite de contexto no prompt

# ---------- Carrega índice + metadados ----------
print("[Llama Del Rey] Carregando índice FAISS e metadados...")
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise RuntimeError(
        "[Llama Del Rey] Index ou metadata não encontrados. "
        "Rode `python indexer.py` primeiro para gerar ./data/faiss.index e ./data/meta.jsonl."
    )

index = faiss.read_index(str(INDEX_PATH))

# Arrays de metadados alinhados com a ordem de inserção no índice
chunks_text: List[str] = []
sources: List[str] = []
source_types: List[str] = []

with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        chunks_text.append(obj.get("text", ""))
        sources.append(obj.get("source_path", ""))
        source_types.append(obj.get("source_type", ""))

print(f"[Llama Del Rey] {len(chunks_text)} chunks carregados.")

# ============================================================
# Esquemas (Pydantic)
# ============================================================

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

class AskDebugResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

# ============================================================
# Funções auxiliares
# ============================================================

def embed_query(text: str) -> List[float]:
    """
    Gera o embedding da pergunta chamando o endpoint de embeddings do Ollama.
    Normaliza L2 o vetor para que a busca por produto interno (FAISS IP) se comporte
    como similaridade do cosseno.
    """
    payload = {"model": EMBED_MODEL, "prompt": text}  # OBS: 'prompt' é o campo esperado pelo Ollama
    resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding") or (data.get("data", [{}])[0].get("embedding"))
    if not vec:
        raise RuntimeError(f"Unexpected embedding response: {data}")
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]

def search_similar(qvec: List[float], top_k: int = TOP_K) -> List[int]:
    """
    Faz a busca vetorial no FAISS e retorna os índices dos melhores resultados.
    """
    q = np.array([qvec], dtype="float32")
    scores, idxs = index.search(q, top_k)
    return [int(i) for i in idxs[0] if i != -1]

# --------------------- Rerank dinâmico (multi-tema) ---------------------

# Stopwords bem básicas em PT; ajuste conforme necessidade
PT_STOP = {
    "de","da","do","das","dos","e","a","o","os","as","um","uma","uns","umas",
    "para","por","com","sem","em","no","na","nos","nas","num","numa",
    "que","ou","se","ao","aos","à","às","pra","pro","cada","entre","sobre",
    "ser","estar","ter","há","tem"
}

def extract_keywords_pt(question: str) -> List[str]:
    """
    Extrai palavras/expressões relevantes da pergunta em PT-BR (multi-tema).
    - Mantém frases entre aspas como uma keyword inteira ("política de férias")
    - Mantém termos com dígitos/símbolos (R$, 13º, ISO9001)
    - Remove stopwords e termos muito curtos
    """
    q = (question or "").strip()
    ql = q.lower()

    # 1) frases entre aspas (“…”, "...") viram keywords inteiras
    quoted_pairs = re.findall(r'"([^"]+)"|“([^”]+)”', q)
    quoted = [a or b for (a, b) in quoted_pairs if (a or b)]
    quoted = [s.strip() for s in quoted if len(s.strip()) > 1]

    # 2) tokens alfanuméricos/símbolos úteis (mantém 13º, iso9001, r$, etc.)
    raw_toks = re.findall(r"[a-zA-ZÀ-ÿ0-9$%º°\-]+", ql)

    # 3) remove stopwords e tokens muito curtos
    toks = [t for t in raw_toks if len(t) > 2 and t not in PT_STOP]

    # 4) junta e deduplica preservando ordem (frases vêm primeiro)
    kws = list(dict.fromkeys(quoted + toks))
    return kws[:10]  # limite de segurança

def keyword_rescue(question: str, limit: int = 3) -> List[int]:
    """
    Resgate por palavras/expressões da pergunta (PT).
    Conta ocorrências simples; barato e eficaz para complementar a busca semântica.
    """
    kws = extract_keywords_pt(question)
    if not kws:
        return []
    scores = []
    for i, txt in enumerate(chunks_text):
        t = txt.lower()
        hit = 0
        for k in kws:
            hit += t.count(k.lower())
        scores.append((hit, i))
    scores.sort(reverse=True)
    return [i for hit, i in scores if hit > 0][:limit]

def rerank_by_keywords(idxs: List[int], keywords: List[str]) -> List[int]:
    """
    Reordena candidatos priorizando trechos que contêm as palavras/expressões da pergunta.
    """
    kw = [k.lower() for k in keywords]
    def score(i: int) -> int:
        t = chunks_text[i].lower()
        s = 0
        for k in kw:
            s += t.count(k)
        return s
    return sorted(idxs, key=score, reverse=True)

# --------------------- Prompt builder (PT-BR apenas) ---------------------

def build_prompt_pt(question: str, contexts: List[str]) -> str:
    """
    Monta o prompt em PT-BR:
    - Empilha trechos até MAX_CONTEXT_CHARS.
    - Instruções para usar somente o CONTEXTO e responder de forma direta.
    """
    # Bloco de contexto com limite de caracteres
    ctx_block = ""
    total = 0
    for i, c in enumerate(contexts, start=1):
        if total + len(c) > MAX_CONTEXT_CHARS:
            break
        ctx_block += f"[CTX {i}]\n{c}\n\n"
        total += len(c)

    system = (
        "Use exclusivamente o CONTEXTO a seguir para responder. "
        "Se a resposta não estiver no contexto, diga que não há informação suficiente. "
        "Responda em português (pt-BR). "
        "Não se apresente. Não faça perguntas de volta. Seja direto e preciso."
    )
    user = (
        f"Pergunta:\n{question}\n\n"
        f"CONTEXTO:\n{ctx_block}"
        "Resposta:"
    )
    return f"{system}\n\n{user}"

def generate_answer(prompt: str) -> str:
    """
    Chama o LLM do Ollama para gerar a resposta final.
    - stream=False para devolver JSON limpo (bom para Postman).
    - temperature baixa para reduzir variação e manter fidelidade ao contexto.
    """
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096
        },
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    if not r.ok:
        raise RuntimeError(f"Ollama generate failed {r.status_code}: {r.text}")
    data = r.json()
    return (data.get("response") or "").strip()

def prepare_context_indices(question: str, qvec: List[float]) -> List[int]:
    """
    Monta a lista final de índices de contexto:
    - Busca semântica no FAISS (Top-K)
    - 'Rescue' por palavras/expressões da própria pergunta
    - Rerank dinâmico priorizando quem contém essas palavras/expressões
    """
    idxs = search_similar(qvec, TOP_K)      # 1) semântica primeiro
    extra = keyword_rescue(question, 3)     # 2) rescue por exatos
    merged = list(dict.fromkeys(extra + idxs))  # 3) merge sem duplicatas

    kws = extract_keywords_pt(question)
    if kws:
        merged = rerank_by_keywords(merged, kws)  # 4) rerank dinâmico

    return merged

# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="Llama Del Rey - Local RAG (PT-BR, multi-tema) com Ollama + FAISS")

@app.get("/health")
def health():
    """Endpoint simples de healthcheck."""
    return {"status": "ok", "chunks": len(chunks_text)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """
    Endpoint principal (PT-BR).
    Corpo esperado:
    {
      "question": "sua pergunta aqui"
    }
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="O campo 'question' é obrigatório.")
    try:
        qvec = embed_query(q)
        idxs = prepare_context_indices(q, qvec)
        contexts = [chunks_text[i] for i in idxs if 0 <= i < len(chunks_text)]
        prompt = build_prompt_pt(q, contexts)
        answer = generate_answer(prompt)
        return AskResponse(answer=answer)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_debug", response_model=AskDebugResponse)
def ask_debug(req: AskRequest):
    """
    Versão com debug: retorna a resposta e as fontes/trechos usados.
    Útil para demos, inspeção e avaliação de qualidade.
    """
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="O campo 'question' é obrigatório.")
    try:
        qvec = embed_query(q)
        idxs = prepare_context_indices(q, qvec)
        contexts = [chunks_text[i] for i in idxs if 0 <= i < len(chunks_text)]
        prompt = build_prompt_pt(q, contexts)
        answer = generate_answer(prompt)

        srcs = []
        for i in idxs[:TOP_K]:
            if 0 <= i < len(chunks_text):
                snippet = chunks_text[i]
                srcs.append({
                    "source_path": sources[i],
                    "source_type": source_types[i],
                    "snippet": (snippet[:300] + ("..." if len(snippet) > 300 else ""))
                })

        return AskDebugResponse(answer=answer, sources=srcs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
