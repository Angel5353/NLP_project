from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    chunking_strategy: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FaissIndexer:
    """
    Build, save, load, and search a FAISS index over text chunks.

    Assumes cosine similarity via normalized embeddings + inner product.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []

    def encode_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into normalized embeddings.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def build(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> None:
        """
        Build FAISS index from chunk dicts.

        Expected chunk format:
            {
                "chunk_id": ...,
                "doc_id": ...,
                "text": ...,
                "start_char": ...,
                "end_char": ...,
                "chunking_strategy": ...
            }
        """
        if not chunks:
            raise ValueError("Cannot build FAISS index from empty chunk list.")

        texts = [c["text"] for c in chunks]
        embeddings = self.encode_texts(texts, batch_size=batch_size)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.metadata = chunks

    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Save FAISS index and metadata.
        """
        if self.index is None:
            raise ValueError("Index has not been built yet.")

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Load FAISS index and metadata from disk.
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search top-k most similar chunks for a query.
        """
        if self.index is None:
            raise ValueError("Index has not been built or loaded.")

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_embedding = self.encode_texts([query], show_progress_bar=False)
        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)

        return results


def build_and_save_index(
    chunks: List[Dict[str, Any]],
    model_name: str,
    index_path: str,
    metadata_path: str,
    batch_size: int = 32,
) -> FaissIndexer:
    """
    Convenience function for one-shot build + save.
    """
    indexer = FaissIndexer(model_name=model_name)
    indexer.build(chunks, batch_size=batch_size)
    indexer.save(index_path=index_path, metadata_path=metadata_path)
    return indexer