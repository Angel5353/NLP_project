from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

from chunking import (
    fixed_size_chunk,
    recursive_chunk,
    build_chunks_for_corpus,
    chunks_to_dicts,
)
from indexing import build_and_save_index


# =========================
# I/O helpers
# =========================

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_json(path: str, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =========================
# Corpus loading
# =========================

def load_corpus_from_json(path: str) -> List[Dict[str, str]]:
    """
    Expected format:
    [
      {"doc_id": "doc1", "text": "..."},
      {"doc_id": "doc2", "text": "..."}
    ]
    """
    docs = load_json(path)

    if not isinstance(docs, list):
        raise ValueError("Corpus JSON must contain a list of documents.")

    validated_docs: List[Dict[str, str]] = []
    for doc in docs:
        if "doc_id" not in doc or "text" not in doc:
            raise ValueError("Each corpus item must contain 'doc_id' and 'text'.")
        validated_docs.append({
            "doc_id": str(doc["doc_id"]),
            "text": str(doc["text"]),
        })

    return validated_docs


def load_corpus_from_jsonl(path: str) -> List[Dict[str, str]]:
    """
    Expected format:
    {"doc_id": "doc1", "text": "..."}
    {"doc_id": "doc2", "text": "..."}
    """
    docs = load_jsonl(path)

    validated_docs: List[Dict[str, str]] = []
    for doc in docs:
        if "doc_id" not in doc or "text" not in doc:
            raise ValueError("Each corpus item must contain 'doc_id' and 'text'.")
        validated_docs.append({
            "doc_id": str(doc["doc_id"]),
            "text": str(doc["text"]),
        })

    return validated_docs


def load_corpus_from_txt_dir(corpus_dir: str) -> List[Dict[str, str]]:
    """
    Load all .txt files from a directory as documents.
    File stem becomes doc_id.
    """
    docs: List[Dict[str, str]] = []
    corpus_path = Path(corpus_dir)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")

    txt_files = sorted(corpus_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in corpus directory: {corpus_dir}")

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        docs.append({
            "doc_id": file_path.stem,
            "text": text,
        })

    return docs


def load_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """
    Auto-detect input format:
    - .json
    - .jsonl
    - directory of .txt files
    """
    path = Path(corpus_path)

    if path.is_dir():
        return load_corpus_from_txt_dir(corpus_path)

    if path.suffix == ".json":
        return load_corpus_from_json(corpus_path)

    if path.suffix == ".jsonl":
        return load_corpus_from_jsonl(corpus_path)

    raise ValueError(
        "Unsupported corpus format. Use a .json file, .jsonl file, or a directory of .txt files."
    )


# =========================
# Stats helpers
# =========================

def summarize_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks:
        return {
            "num_chunks": 0,
            "avg_chunk_length": 0.0,
            "min_chunk_length": 0,
            "max_chunk_length": 0,
        }

    lengths = [len(c["text"]) for c in chunks]

    return {
        "num_chunks": len(chunks),
        "avg_chunk_length": sum(lengths) / len(lengths),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }


# =========================
# Build flows
# =========================

def build_fixed_chunks(
    docs: List[Dict[str, str]],
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    chunks = build_chunks_for_corpus(
        docs=docs,
        chunk_fn=fixed_size_chunk,
        chunking_strategy="fixed",
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunks_to_dicts(chunks)


def build_recursive_chunks(
    docs: List[Dict[str, str]],
    chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    chunks = build_chunks_for_corpus(
        docs=docs,
        chunk_fn=recursive_chunk,
        chunking_strategy="recursive",
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunks_to_dicts(chunks)


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk files and FAISS indexes.")

    # input
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to corpus JSON / JSONL, or directory of .txt files.",
    )

    # model
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model used to build FAISS indexes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )

    # fixed chunk params
    parser.add_argument(
        "--fixed_chunk_size",
        type=int,
        default=500,
        help="Fixed chunk size in characters.",
    )
    parser.add_argument(
        "--fixed_overlap",
        type=int,
        default=100,
        help="Fixed chunk overlap in characters.",
    )

    # recursive chunk params
    parser.add_argument(
        "--recursive_chunk_size",
        type=int,
        default=500,
        help="Recursive chunk size in characters.",
    )
    parser.add_argument(
        "--recursive_overlap",
        type=int,
        default=100,
        help="Recursive chunk overlap in characters.",
    )

    # output root
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="Root directory for chunks, indexes, and summary.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    indexes_dir = output_dir / "indexes"
    summary_path = output_dir / "build_index_summary.json"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    indexes_dir.mkdir(parents=True, exist_ok=True)

    # 1) load corpus
    docs = load_corpus(args.corpus_path)
    print(f"Loaded {len(docs)} documents.")

    # 2) build fixed chunks
    print("Building fixed-size chunks...")
    fixed_chunks = build_fixed_chunks(
        docs=docs,
        chunk_size=args.fixed_chunk_size,
        overlap=args.fixed_overlap,
    )
    fixed_chunks_path = chunks_dir / "fixed_chunks.jsonl"
    save_jsonl(str(fixed_chunks_path), fixed_chunks)
    print(f"Saved fixed chunks to {fixed_chunks_path}")

    # 3) build recursive chunks
    print("Building recursive chunks...")
    recursive_chunks = build_recursive_chunks(
        docs=docs,
        chunk_size=args.recursive_chunk_size,
        overlap=args.recursive_overlap,
    )
    recursive_chunks_path = chunks_dir / "recursive_chunks.jsonl"
    save_jsonl(str(recursive_chunks_path), recursive_chunks)
    print(f"Saved recursive chunks to {recursive_chunks_path}")

    # 4) build fixed FAISS index
    print("Building fixed FAISS index...")
    fixed_index_path = indexes_dir / "fixed_index.faiss"
    fixed_metadata_path = indexes_dir / "fixed_metadata.pkl"
    build_and_save_index(
        chunks=fixed_chunks,
        model_name=args.embedding_model,
        index_path=str(fixed_index_path),
        metadata_path=str(fixed_metadata_path),
        batch_size=args.batch_size,
    )
    print(f"Saved fixed index to {fixed_index_path}")
    print(f"Saved fixed metadata to {fixed_metadata_path}")

    # 5) build recursive FAISS index
    print("Building recursive FAISS index...")
    recursive_index_path = indexes_dir / "recursive_index.faiss"
    recursive_metadata_path = indexes_dir / "recursive_metadata.pkl"
    build_and_save_index(
        chunks=recursive_chunks,
        model_name=args.embedding_model,
        index_path=str(recursive_index_path),
        metadata_path=str(recursive_metadata_path),
        batch_size=args.batch_size,
    )
    print(f"Saved recursive index to {recursive_index_path}")
    print(f"Saved recursive metadata to {recursive_metadata_path}")

    # 6) save summary
    summary = {
        "corpus_path": args.corpus_path,
        "num_documents": len(docs),
        "embedding_model": args.embedding_model,
        "batch_size": args.batch_size,
        "fixed": {
            "chunk_size": args.fixed_chunk_size,
            "overlap": args.fixed_overlap,
            "chunks_path": str(fixed_chunks_path),
            "index_path": str(fixed_index_path),
            "metadata_path": str(fixed_metadata_path),
            "stats": summarize_chunks(fixed_chunks),
        },
        "recursive": {
            "chunk_size": args.recursive_chunk_size,
            "overlap": args.recursive_overlap,
            "chunks_path": str(recursive_chunks_path),
            "index_path": str(recursive_index_path),
            "metadata_path": str(recursive_metadata_path),
            "stats": summarize_chunks(recursive_chunks),
        },
    }

    save_json(str(summary_path), summary)
    print(f"Saved summary to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()