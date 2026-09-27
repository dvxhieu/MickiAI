"""Long-term memory backed by ChromaDB (semantic search / RAG)."""

from __future__ import annotations

from typing import Any

from config.settings import settings


class VectorMemory:
    """Persistent vector store for agent memory.

    Documents are stored with their embeddings, so we can later retrieve the
    most semantically relevant chunks for a given query (RAG).
    """

    def __init__(self, path: str | None = None) -> None:
        import chromadb

        self.path = path or settings.chroma_path
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection("agent_memory")

    # ------------------------------------------------------------------
    def add(self, text: str, metadata: dict, doc_id: str) -> None:
        """Store a document (text + metadata) in the collection."""
        self.collection.add(documents=[text], metadatas=[metadata], ids=[doc_id])

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int | None = None) -> dict[str, Any]:
        """Semantic search — return the closest documents to ``query``."""
        return self.collection.query(
            query_texts=[query], n_results=top_k or settings.memory_top_k
        )

    # ------------------------------------------------------------------
    def search_context(self, query: str, top_k: int | None = None) -> str:
        """Return a human-readable context string built from search results."""
        results = self.search(query, top_k)
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return ""
        lines = []
        for doc, meta in zip(docs, metas):
            role = meta.get("role", "unknown")
            lines.append(f"[{role}] {doc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection("agent_memory")
        self.collection = self.client.get_or_create_collection("agent_memory")