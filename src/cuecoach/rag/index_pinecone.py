from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


# -----------------------------
# Defaults / paths
# -----------------------------
DEFAULT_CHUNKS_DIR = Path("data/chunks")  # expects typed/, ocr/ subfolders but will scan recursively
DEFAULT_NAMESPACE = "default"

# OpenAI embedding model defaults (override via env OPENAI_EMBED_MODEL)
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Known dimensions (to create index correctly)
EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


# -----------------------------
# Env helpers
# -----------------------------
def _load_env() -> None:
    """
    Why:
    - You store keys in cuecoach/.env (repo root). `load_dotenv()` reads that file so
      os.environ contains OPENAI_API_KEY / PINECONE_API_KEY / etc.
    """
    load_dotenv()  # loads .env from current working directory if present


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, default)


# -----------------------------
# Pinecone metadata sanitation
# -----------------------------
def pinecone_safe_value(v: Any) -> Any:
    """
    Pinecone metadata values must be:
    - string, number, boolean, or list of strings
    It rejects null/None (your current crash), and rejects list of non-strings.
    """
    if v is None:
        return ""  # critical fix: Pinecone rejects null
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [str(x) for x in v]
    # Anything else (dicts, objects) -> string
    return str(v)


def pinecone_safe_metadata(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: pinecone_safe_value(v) for k, v in d.items()}


# -----------------------------
# Chunk loading
# -----------------------------
def iter_jsonl_files(chunks_dir: Path) -> Iterable[Path]:
    yield from sorted(chunks_dir.rglob("*.jsonl"))


def load_chunks_from_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# -----------------------------
# OpenAI embeddings
# -----------------------------
def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    """
    Why:
    - Batch embedding is faster and cheaper than per-chunk calls.
    """
    resp = client.embeddings.create(model=model, input=texts)
    # OpenAI returns same order as input
    return [item.embedding for item in resp.data]


# -----------------------------
# Pinecone index setup
# -----------------------------
def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> None:
    """
    Why:
    - Create index if missing, otherwise reuse.
    - Dimension must match embedding dimension.
    """
    existing = {i["name"] for i in pc.list_indexes()}
    if index_name in existing:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


# -----------------------------
# Vector building
# -----------------------------
def chunk_to_vector_payload(
    chunk: Dict[str, Any],
    embedding: List[float],
) -> Dict[str, Any]:
    """
    Why:
    - Keep metadata small and Pinecone-compliant.
    - Do NOT send nulls (fixed via pinecone_safe_metadata).
    """
    raw_meta = {
        "doc_id": chunk.get("doc_id"),
        "source": chunk.get("source"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),  # may be None -> will become ""
        "url": chunk.get("url"),          # may be None -> will become ""
        "topic": chunk.get("topic"),
        "skill_level": chunk.get("skill_level"),
    }

    # Optional: store text in metadata (useful for debugging; can increase metadata size).
    # Uncomment if you want it:
    # raw_meta["text"] = chunk.get("text", "")

    meta = pinecone_safe_metadata(raw_meta)

    return {
        "id": str(chunk["chunk_id"]),
        "values": embedding,
        "metadata": meta,
    }


def batched(iterable: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Index chunk JSONL into Pinecone with OpenAI embeddings.")
    parser.add_argument("--chunks-dir", default=str(DEFAULT_CHUNKS_DIR), help="Directory containing chunk .jsonl files.")
    parser.add_argument("--namespace", default=optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE), help="Pinecone namespace.")
    parser.add_argument("--embed-model", default=optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL), help="OpenAI embedding model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Chunks per embedding request.")
    parser.add_argument("--upsert-batch-size", type=int, default=100, help="Vectors per Pinecone upsert.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of JSONL files (0 = no limit).")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        raise SystemExit(f"Chunks dir not found: {chunks_dir}")

    # Required env vars (kept in script as requested)
    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index_name = require_env("PINECONE_INDEX")

    # Optional env vars with safe defaults
    metric = optional_env("PINECONE_METRIC", "cosine")
    cloud = optional_env("PINECONE_CLOUD", "aws")
    region = optional_env("PINECONE_REGION", "us-east-1")

    embed_model = args.embed_model
    if embed_model not in EMBED_DIMS:
        raise SystemExit(
            f"Unknown embedding model '{embed_model}'. Add it to EMBED_DIMS with the correct dimension."
        )
    dimension = EMBED_DIMS[embed_model]

    # Init clients
    oai = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)

    ensure_index(
        pc=pc,
        index_name=pinecone_index_name,
        dimension=dimension,
        metric=metric,
        cloud=cloud,
        region=region,
    )

    index = pc.Index(pinecone_index_name)

    # Gather JSONL files
    jsonl_files = list(iter_jsonl_files(chunks_dir))
    if args.max_files and args.max_files > 0:
        jsonl_files = jsonl_files[: args.max_files]

    if not jsonl_files:
        raise SystemExit(f"No .jsonl found under: {chunks_dir}")

    total_vectors = 0
    total_files = 0

    for jf in jsonl_files:
        total_files += 1

        # Load all chunks from this file
        chunks = list(load_chunks_from_jsonl(jf))
        if not chunks:
            print(f"Skip empty JSONL: {jf}")
            continue

        # Embed in batches
        vectors: List[Dict[str, Any]] = []

        for chunk_batch in batched(chunks, args.batch_size):
            texts = [c.get("text", "") for c in chunk_batch]
            # empty text safety
            texts = [t if t else " " for t in texts]

            embeddings = embed_texts(oai, embed_model, texts)

            for c, emb in zip(chunk_batch, embeddings):
                vectors.append(chunk_to_vector_payload(c, emb))

        # Upsert in batches
        for vbatch in batched(vectors, args.upsert_batch_size):
            # This is where your previous crash happened (metadata had None).
            # Now it won't.
            index.upsert(vectors=vbatch, namespace=args.namespace)

        total_vectors += len(vectors)
        print(f"Indexed: {jf.relative_to(chunks_dir)} -> {len(vectors)} vectors")

    print(f"Done. Files: {total_files}, Total vectors: {total_vectors}, Namespace: {args.namespace}")


if __name__ == "__main__":
    main()