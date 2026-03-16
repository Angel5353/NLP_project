from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Dict, Any, Optional


@dataclass
class Chunk:
    """
    A retrieval chunk produced from a source document.
    """
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    chunking_strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size character chunks with overlap.

    Returns a list of dicts with:
        - text
        - start_char
        - end_char
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[Dict[str, Any]] = []
    n = len(text)

    if n == 0:
        return chunks

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text[start:end]

        chunks.append({
            "text": chunk_text,
            "start_char": start,
            "end_char": end,
        })

        if end == n:
            break

        start = end - overlap

    return chunks


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    separators: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Split text using LangChain's RecursiveCharacterTextSplitter.

    This tries to preserve paragraph / sentence boundaries better than
    fixed-size chunking.

    Returns a list of dicts with:
        - text
        - start_char
        - end_char
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as e:
        raise ImportError(
            "recursive_chunk requires langchain-text-splitters. "
            "Install it with: pip install langchain-text-splitters"
        ) from e

    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators,
    )

    split_texts = splitter.split_text(text)

    chunks: List[Dict[str, Any]] = []
    cursor = 0

    # We try to recover approximate start/end offsets from the original text.
    # This is useful for later evaluation and debugging.
    for piece in split_texts:
        idx = text.find(piece, cursor)
        if idx == -1:
            # fallback: if exact find fails, use cursor as approximate location
            idx = cursor

        start_char = idx
        end_char = idx + len(piece)

        chunks.append({
            "text": piece,
            "start_char": start_char,
            "end_char": end_char,
        })

        cursor = end_char

    return chunks


def build_chunks_for_document(
    doc_id: str,
    text: str,
    chunk_fn: Callable[..., List[Dict[str, Any]]],
    chunking_strategy: str,
    **chunk_kwargs: Any,
) -> List[Chunk]:
    """
    Chunk a single document and attach metadata.
    """
    raw_chunks = chunk_fn(text, **chunk_kwargs)

    output: List[Chunk] = []
    for i, ch in enumerate(raw_chunks):
        output.append(
            Chunk(
                chunk_id=f"{doc_id}_chunk_{i}",
                doc_id=doc_id,
                text=ch["text"],
                start_char=ch["start_char"],
                end_char=ch["end_char"],
                chunking_strategy=chunking_strategy,
            )
        )
    return output


def build_chunks_for_corpus(
    docs: Iterable[Dict[str, str]],
    chunk_fn: Callable[..., List[Dict[str, Any]]],
    chunking_strategy: str,
    **chunk_kwargs: Any,
) -> List[Chunk]:
    """
    Chunk all documents in the corpus.

    Expected input format for docs:
        {"doc_id": "...", "text": "..."}
    """
    all_chunks: List[Chunk] = []

    for doc in docs:
        doc_id = doc["doc_id"]
        text = doc["text"]
        doc_chunks = build_chunks_for_document(
            doc_id=doc_id,
            text=text,
            chunk_fn=chunk_fn,
            chunking_strategy=chunking_strategy,
            **chunk_kwargs,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks


def chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    """
    Convert Chunk objects to plain dicts for JSON serialization.
    """
    return [c.to_dict() for c in chunks]