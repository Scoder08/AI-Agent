# utils/index_faiss.py  –  fast FAISS index builder with batched + parallel embedding
from __future__ import annotations

import asyncio, os, pickle, glob, ast, pathlib
from typing import Iterable, List

import numpy as np, faiss, tiktoken

# --------------------------------------------------------------------- #
# 0)  Imports / fall-back for async embeddings
# --------------------------------------------------------------------- #
try:
    from langchain_openai import AsyncOpenAIEmbeddings as _Emb
    ASYNC = True
except ImportError:
    from langchain_openai.embeddings import OpenAIEmbeddings as _Emb
    ASYNC = False

# --------------------------------------------------------------------- #
# 1)  Config
# --------------------------------------------------------------------- #
KEEP_DIRS   = {"utils", "pulse_utils", "mobile", "worker", "crons", "internal_tools_api"}
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE  = 64          # snippets per request  (keep under ~8K tokens)
PARALLEL    = 8           # concurrent requests

ROOT        = pathlib.Path(__file__).resolve().parents[1]  # repo root
INDEX_PATH  = ROOT / "data" / "vec_faiss.pkl"
INDEX_PATH.parent.mkdir(exist_ok=True, parents=True)

TOKENIZER   = tiktoken.get_encoding("cl100k_base")
EMBEDDER    = _Emb(model=EMBED_MODEL)

# --------------------------------------------------------------------- #
# 2)  Generator: yield Python function / class bodies
# --------------------------------------------------------------------- #
def iter_py_snippets(root: pathlib.Path = ROOT) -> Iterable[str]:
    for path in glob.glob(str(root / "**" / "*.py"), recursive=True):
        rel = pathlib.Path(path).relative_to(root)
        if rel.parts[0] not in KEEP_DIRS:
            continue
        if any(part in ("tests", "migrations", "venv") for part in rel.parts):
            continue

        src = pathlib.Path(path).read_text("utf-8", errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue  # skip invalid files

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start, end = node.lineno - 1, node.end_lineno
                yield "\n".join(src.splitlines()[start:end])

# --------------------------------------------------------------------- #
# 3)  Async helpers
# --------------------------------------------------------------------- #
async def _embed_documents_snippets(snips: List[str]) -> List[List[float]]:
    """Handle both async and sync embedding classes transparently."""
    if ASYNC:
        return await EMBEDDER.embed_documents(snips)  # type: ignore[arg-type]
    # sync fallback — run in thread to avoid blocking
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, EMBEDDER.embed_documents, snips)  # type: ignore[arg-type]

async def _embed_query(text: str) -> List[float]:
    if ASYNC:
        return await EMBEDDER.embed_query(text)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, EMBEDDER.embed_query, text)

# --------------------------------------------------------------------- #
# 4)  Build FAISS index  (only runs if pickle absent)
# --------------------------------------------------------------------- #
async def _build_index():
    print("🔄  Building FAISS index (batched / parallel)…")
    snippets, vectors = [], []
    batch, tasks = [], []
    sem = asyncio.Semaphore(PARALLEL)

    async def schedule(batch_snips: List[str], bid: int):
        async with sem:
            vecs = await _embed_documents_snippets(batch_snips)
            vectors.extend(vecs)
            if bid % 20 == 0:
                print(f"   ✓ finished batch {bid}")

    batch_id = 0
    for snip_id, snippet in enumerate(iter_py_snippets(ROOT), 1):
        # truncate super-long snippets to fit token limit
        if len(TOKENIZER.encode(snippet)) > 8000:
            snippet = TOKENIZER.decode(TOKENIZER.encode(snippet)[:8000])
        snippets.append(snippet)
        batch.append(snippet)

        if len(batch) == BATCH_SIZE:
            batch_id += 1
            tasks.append(asyncio.create_task(schedule(batch.copy(), batch_id)))
            batch.clear()

        if snip_id % 500 == 0:
            print(f"   · queued {snip_id} snippets")

    if batch:
        batch_id += 1
        tasks.append(asyncio.create_task(schedule(batch.copy(), batch_id)))

    await asyncio.gather(*tasks)
    print(f"✅  Embedded {len(snippets)} snippets total")

    vec_np = np.asarray(vectors, dtype="float32")
    index  = faiss.IndexFlatIP(vec_np.shape[1])
    index.add(vec_np)

    with INDEX_PATH.open("wb") as f:
        pickle.dump((index, snippets), f)
    print(f"🔐  Saved index → {INDEX_PATH}")

# --------------------------------------------------------------------- #
# 5)  Load or build at import time
# --------------------------------------------------------------------- #
if INDEX_PATH.exists():
    print(f"🔹  Loading FAISS index from {INDEX_PATH}")
    faiss_index, meta = pickle.load(INDEX_PATH.open("rb"))
else:
    asyncio.run(_build_index())
    faiss_index, meta = pickle.load(INDEX_PATH.open("rb"))

# --------------------------------------------------------------------- #
# 6)  Public search API
# --------------------------------------------------------------------- #
async def vector_search(text: str, k: int = 3):
    """
    Async-friendly search that works whether we’re using the asynchronous
    or synchronous LangChain embedding class.
    """
    loop = asyncio.get_running_loop()

    if ASYNC:
        emb = await _embed_query(text)                    # true async path
    else:                                                 # sync fallback
        emb = await loop.run_in_executor(
            None,                                         # default ThreadPool
            EMBEDDER.embed_query,                         # blocking fn
            text,
        )

    vec = np.asarray([emb], dtype="float32")
    _, idx = faiss_index.search(vec, k)
    return [meta[i] for i in idx[0] if i >= 0]