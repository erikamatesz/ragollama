"""
Llama Del Rey - Indexer
-----------------------
Este script percorre a pasta ./docs, lê PDFs e CSVs, divide o conteúdo em
chunks, gera embeddings no Ollama e salva um índice FAISS em ./data.
Também cria um meta.jsonl com os metadados de cada chunk.
"""

# ============================================================
# 1) Imports e configuração básica
# ============================================================
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

# Carrega variáveis do .env se existir (não quebra se faltar)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# 2) Configurações e caminhos (env, modelos, limites)
# ============================================================
DOCS_DIR = Path("docs")             # Onde ficam os PDFs/CSVs de entrada
DATA_DIR = Path("data")             # Saída (índice + metadados)
DATA_DIR.mkdir(exist_ok=True)
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.jsonl"

# Endpoints/modelos do Ollama (podem vir do .env)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Tamanho dos blocos de texto (chunks) e sobreposição entre eles (para PDF)
# -----------------------------------------------------------------
# CHUNK_SIZE = quantos caracteres vão em cada pedaço (chunk)
# CHUNK_OVERLAP = quantos caracteres do final do chunk anterior
#                 serão repetidos no início do próximo chunk
# Ex.: com CHUNK_SIZE=10 e OVERLAP=3, parte do fim do chunk 1
# reaparece no começo do chunk 2, preservando contexto.
# -----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

CSV_MAX_ROWS = int(os.getenv("CSV_MAX_ROWS", 200_000))  # segurança p/ CSV grande

# ============================================================
# 3) Utilidades de chunking e leitura de fontes
# ============================================================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Divide um texto longo em pedaços (chunks) de tamanho fixo com sobreposição.

    - Normaliza espaços em branco para evitar quebras estranhas.
    - Usa janela deslizante com `size` e recuo `overlap` para manter contexto entre chunks.

    Retorna:
        Lista de strings (cada item é um chunk).
    """
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
        # recua para criar sobreposição com o próximo chunk
        start = max(0, end - overlap)
    return chunks


def read_pdf(path: Path) -> str:
    """
    Extrai texto de um PDF usando pypdf.
    Em caso de erro, registra um aviso e retorna string vazia.
    """
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        print(f"[Llama Del Rey] WARN: Falha ao ler PDF {path}: {e}")
        return ""


def iter_csv_rows_pt(path: Path, max_rows: int = CSV_MAX_ROWS) -> List[str]:
    """
    Lê um CSV com cabeçalhos em PT-BR e transforma CADA LINHA em UM TEXTO curto.

    Espera colunas: Nome, Username, Cargo, Senioridade.
    O texto gerado é autoexplicativo e estável para busca:
      "Nome: {..} | Username: {..} | Cargo/Role: {..} | Senioridade/Level: {..}"
    """
    rows = []
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                r = csv.DictReader(f)
                if not r.fieldnames:
                    print(f"[Llama Del Rey] WARN: CSV sem cabeçalho: {path}")
                    return rows
                for i, row in enumerate(r):
                    if i >= max_rows:
                        break
                    nome        = (row.get("Nome") or "").strip()
                    username    = (row.get("Username") or "").strip()
                    cargo       = (row.get("Cargo") or "").strip()
                    senioridade = (row.get("Senioridade") or "").strip()
                    if not (nome or username or cargo or senioridade):
                        continue
                    text = (
                        f"Nome: {nome} | Username: {username} | "
                        f"Cargo/Role: {cargo} | Senioridade/Level: {senioridade}"
                    )
                    rows.append(text)
            break  # leu com sucesso, sai do loop de encodings
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"[Llama Del Rey] WARN: Falha ao ler CSV {path}: {e}")
            break
    return rows

# ============================================================
# 4) Embeddings (Ollama) e normalização L2
# ============================================================

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Gera embeddings para uma lista de textos chamando o Ollama (/api/embeddings).

    Observações:
    - Trabalha item a item.
    - Extrai o vetor tanto do campo "embedding" quanto de "data[0].embedding",
      cobrindo os formatos mais comuns de retorno.

    Exemplo:
        >>> embed_texts(["um cachorro correndo", "uma banana amarela"])
        [
            [0.12, -0.34, 0.88, ...],
            [0.45, 0.67, -0.12, ...]
        ]

    Retorna:
        Lista de vetores (lista de floats).
    """
    vectors = []
    for t in texts:
        try:
            payload = {"model": EMBED_MODEL, "prompt": t}
            resp = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding") or (data.get("data", [{}])[0].get("embedding"))
            if not vec:
                print(f"[Llama Del Rey] WARN: Resposta de embedding inesperada para texto: {t[:80]}...")
                continue
            vectors.append(vec)
        except Exception as e:
            print(f"[Llama Del Rey] WARN: embedding falhou ({e}); pulando chunk.")
    return vectors


def l2_normalize(vecs: List[List[float]]) -> List[List[float]]:
    """
    Deixa todos os vetores com o mesmo "tamanho" (comprimento 1).
    Isso faz o IndexFlatIP (produto interno) se comportar como similaridade do cosseno.
    """
    out = []
    for v in vecs:
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out

# ============================================================
# 5) Varredura de documentos → produção de chunks com metadados
# ============================================================

def iter_docs(doc_dir: Path) -> List[Tuple[str, str, int, str, str]]:
    """
    Percorre a pasta de documentos e produz tuplas com os chunks.

    Retorna uma lista de tuplas no formato:
      (doc_id, chunk_text, chunk_index, source_path, source_type)

    - doc_id: UUID do chunk (útil para rastrear)
    - chunk_text: texto do chunk
    - chunk_index: posição do chunk dentro do arquivo
    - source_path: caminho do arquivo de origem
    - source_type: "pdf" ou "csv"
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
            if not full_text.strip():
                print(f"[Llama Del Rey] WARN: Sem texto em {path}, ignorando.")
                continue
            chunks = chunk_text(full_text)
            print(f"[Llama Del Rey] {path.name}: {len(chunks)} chunks (PDF)")
            for idx, ch in enumerate(chunks):
                if not ch.strip():
                    continue
                doc_id = str(uuid.uuid4())
                items.append((doc_id, ch, idx, str(path), "pdf"))

        else:  # .csv —> 1 linha = 1 chunk
            row_texts = iter_csv_rows_pt(path)
            if not row_texts:
                print(f"[Llama Del Rey] WARN: Sem linhas úteis em {path}, ignorando.")
                continue
            print(f"[Llama Del Rey] {path.name}: {len(row_texts)} chunks (CSV linhas)")
            for idx, ch in enumerate(row_texts):
                doc_id = str(uuid.uuid4())
                items.append((doc_id, ch, idx, str(path), "csv"))

    return items

# ============================================================
# 6) Pipeline principal (gera vetores, monta índice, salva)
# ============================================================

def main():
    """
    Pipeline completo:
    - Coleta chunks dos documentos (PDF por caracteres; CSV por linha)
    - Gera embeddings (Ollama)
    - Normaliza vetores
    - Cria índice FAISS (IndexFlatIP)
    - Salva índice e metadados alinhados
    """
    print("[Llama Del Rey] Scanning docs...")
    triples = iter_docs(DOCS_DIR)
    if not triples:
        print("[Llama Del Rey] No content found in docs/. Exiting.")
        return

    print(f"[Llama Del Rey] {len(triples)} chunks to embed...")
    texts = [t[1] for t in triples]

    # 1) Embeddings via Ollama
    vecs = embed_texts(texts)
    if not vecs:
        print("[Llama Del Rey] No embeddings generated. Exiting.")
        return

    # 2) Normalização L2
    vecs = l2_normalize(vecs)

    # 3) Criação do índice FAISS
    dim = len(vecs[0])
    index = faiss.IndexFlatIP(dim)

    # 4) Adiciona todos os vetores ao índice
    mat = np.array(vecs, dtype="float32")
    index.add(mat)

    # 5) Salva o índice em disco
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[Llama Del Rey] Saved FAISS index -> {INDEX_PATH}")

    # 6) Salva metadados (mesma ordem dos vetores no índice!)
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
