from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_NAMESPACE = "default"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

# IMPORTANT:
# Pinecone fetch can fail with 431 if the request becomes too large.
# Your IDs are long. Keep this small.
DEFAULT_FETCH_BATCH_SIZE = 10


# -----------------------------
# Env helpers
# -----------------------------
def _load_env() -> None:
    # Loads cuecoach/.env when you run from repo root
    load_dotenv(dotenv_path=Path(".env"), override=False)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required env var: {name}")
    return value


def require_any_env(names: List[str]) -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    raise SystemExit(f"Missing required env var: one of {', '.join(names)}")


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, default)


# -----------------------------
# Pinecone metadata sanitation
# -----------------------------
def pinecone_safe_value(v: Any) -> Any:
    # Pinecone metadata values must be: string, number, boolean, or list of strings.
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, list):
        return [str(x) for x in v]
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


def batched(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# -----------------------------
# OpenAI embeddings
# -----------------------------
def embed_texts(client: OpenAI, model: str, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
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
    # Serverless index create only if missing
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
# Update-only helpers
# -----------------------------
def existing_ids_in_pinecone(
    index: Any,
    ids: List[str],
    namespace: str,
    *,
    fetch_batch_size: int,
) -> Set[str]:
    """
    Pinecone fetch returns only vectors that exist.
    We use that to filter updates so we don't create new vectors.
    """
    out: Set[str] = set()

    # Keep fetch batch size small to avoid 431 (Request Header Fields Too Large)
    for id_batch in batched(ids, fetch_batch_size):
        resp = index.fetch(ids=id_batch, namespace=namespace)
        found = getattr(resp, "vectors", None) or {}
        out.update(found.keys())

    return out


# -----------------------------
# Vector building
# -----------------------------
def chunk_to_vector_payload(
    chunk: Dict[str, Any],
    embedding: List[float],
    *,
    text_max_chars: int,
) -> Dict[str, Any]:
    text = (chunk.get("text") or "").strip()
    if not text:
        text = " "  # embedding API requires a non-empty string
    text_snippet = text[:text_max_chars]

    raw_meta = {
        "doc_id": chunk.get("doc_id"),
        "source": chunk.get("source"),
        "title": chunk.get("title"),
        "section": chunk.get("section"),
        "url": chunk.get("url"),
        "topic": chunk.get("topic"),
        "skill_level": chunk.get("skill_level"),
        # Store text in metadata so ask.py can answer without separate storage
        "text": text_snippet,
    }

    meta = pinecone_safe_metadata(raw_meta)

    return {
        "id": str(chunk["chunk_id"]),
        "values": embedding,
        "metadata": meta,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Update Pinecone vectors/metadata from chunk JSONL using OpenAI embeddings."
    )
    parser.add_argument(
        "--chunks-dir",
        default=str(DEFAULT_CHUNKS_DIR),
        help="Directory containing chunk .jsonl files (scanned recursively).",
    )
    parser.add_argument(
        "--namespace",
        default=optional_env("PINECONE_NAMESPACE", DEFAULT_NAMESPACE),
        help="Pinecone namespace.",
    )
    parser.add_argument(
        "--embed-model",
        default=optional_env("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        help="OpenAI embedding model.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Chunks per embedding request.")
    parser.add_argument("--upsert-batch-size", type=int, default=100, help="Vectors per Pinecone upsert.")
    parser.add_argument("--max-files", type=int, default=0, help="Limit number of JSONL files (0 = no limit).")
    parser.add_argument(
        "--text-max-chars",
        type=int,
        default=3000,
        help="Max chars of chunk text stored in Pinecone metadata.",
    )
    parser.add_argument(
        "--fetch-batch-size",
        type=int,
        default=DEFAULT_FETCH_BATCH_SIZE,
        help="How many IDs to send per Pinecone fetch (keep small to avoid 431).",
    )
    parser.add_argument(
        "--allow-new",
        action="store_true",
        help="Allow creating new vectors if they don't exist already (default is update-only).",
    )
    args = parser.parse_args()

    # Default behavior: UPDATE-ONLY unless --allow-new is provided
    update_only = not args.allow_new

    chunks_dir = Path(args.chunks_dir)
    if not chunks_dir.exists():
        raise SystemExit(f"Chunks dir not found: {chunks_dir}")

    # Required env vars (kept in script)
    openai_key = require_env("OPENAI_API_KEY")
    pinecone_key = require_env("PINECONE_API_KEY")
    pinecone_index_name = require_any_env(["PINECONE_INDEX", "PINECONE_INDEX_NAME"])

    metric = optional_env("PINECONE_METRIC", "cosine")
    cloud = optional_env("PINECONE_CLOUD", "aws")
    region = optional_env("PINECONE_REGION", "us-east-1")

    embed_model = args.embed_model
    if embed_model not in EMBED_DIMS:
        raise SystemExit(
            f"Unknown embedding model '{embed_model}'. Add it to EMBED_DIMS with the correct dimension."
        )
    dimension = EMBED_DIMS[embed_model]

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

    jsonl_files = list(iter_jsonl_files(chunks_dir))
    if args.max_files and args.max_files > 0:
        jsonl_files = jsonl_files[: args.max_files]

    if not jsonl_files:
        raise SystemExit(f"No .jsonl found under: {chunks_dir}")

    total_updated = 0
    total_skipped_new = 0
    total_files = 0

    for jf in jsonl_files:
        total_files += 1
        chunks = list(load_chunks_from_jsonl(jf))
        if not chunks:
            continue

        ids = [str(c["chunk_id"]) for c in chunks if "chunk_id" in c]
        if not ids:
            continue

        if update_only:
            exists = existing_ids_in_pinecone(
                index,
                ids,
                args.namespace,
                fetch_batch_size=max(1, args.fetch_batch_size),
            )

            if not exists:
                total_skipped_new += len(ids)
                print(f"Skip (no existing ids found): {jf.relative_to(chunks_dir)}")
                continue

            filtered_chunks = [c for c in chunks if str(c.get("chunk_id")) in exists]
            skipped = len(chunks) - len(filtered_chunks)
            if skipped:
                total_skipped_new += skipped
            chunks = filtered_chunks

        if not chunks:
            continue

        vectors: List[Dict[str, Any]] = []

        for chunk_batch in batched(chunks, args.batch_size):
            texts = [(c.get("text") or "").strip() for c in chunk_batch]
            texts = [t if t else " " for t in texts]

            embeddings = embed_texts(oai, embed_model, texts)

            for c, emb in zip(chunk_batch, embeddings):
                vectors.append(
                    chunk_to_vector_payload(
                        c,
                        emb,
                        text_max_chars=args.text_max_chars,
                    )
                )

        for vbatch in batched(vectors, args.upsert_batch_size):
            index.upsert(vectors=vbatch, namespace=args.namespace)

        total_updated += len(vectors)
        print(f"Updated: {jf.relative_to(chunks_dir)} -> {len(vectors)} vectors")

    mode = "UPDATE-ONLY" if update_only else "ALLOW-NEW"
    print(
        f"Done ({mode}). Files scanned: {total_files}. "
        f"Updated: {total_updated} vectors. "
        f"Skipped (would-be new): {total_skipped_new} vectors. "
        f"Namespace: {args.namespace}"
    )


if __name__ == "__main__":
    main()
