"""
rag_engine.py — BM25 Keyword Search Knowledge Base
Indexes only the `description:` prose block from each YAML policy file.
Structured eligibility fields (billing codes, annual max, etc.) are parsed
by policy_loader.py for the simulation — RAG sees only human-readable prose.
Billing codes are read directly from the YAML field for exact-match boosting.
ChromaDB and OpenAI embeddings are not used — BM25 is sufficient for 10 documents.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import yaml
import bm25s
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Billing code exact-match detection (query-side only)
# ---------------------------------------------------------------------------

_BILLING_CODE_PATTERN = re.compile(r"\b([A-Z][0-9A-Z]{2,6})\b")


def _query_contains_billing_code(query: str) -> Optional[str]:
    """
    Return the first billing code found in the query string, or None.
    Used to trigger exact-match priority boosting in BM25 ranking.
    """
    codes = set(_BILLING_CODE_PATTERN.findall(query.upper()))
    return next(iter(codes), None) if codes else None


# ---------------------------------------------------------------------------
# Core knowledge base
# ---------------------------------------------------------------------------

class HighPrecisionKnowledgeBase:
    """
    BM25 keyword retriever over YAML policy description blocks.

    Only the `description:` prose field from each *.yaml file is indexed.
    Structured eligibility fields are NOT in the RAG corpus — they belong
    to the simulation engine, not the language model.

    Billing codes come directly from the YAML `eligible_billing_codes` field,
    enabling exact-match boosting without regex extraction from prose.
    When the query contains a billing code (e.g. "J9331"), documents whose
    eligible_billing_codes list contains that code are forced to rank first,
    ensuring the most relevant policy surfaces above semantically-similar ones.
    """

    def __init__(
        self,
        docs_dirs: list[str | Path] | str | Path,
        top_k: int = 3,
    ) -> None:
        # Accept either a single path or a list of paths
        if isinstance(docs_dirs, (str, Path)):
            self.docs_dirs = [Path(docs_dirs)]
        else:
            self.docs_dirs = [Path(d) for d in docs_dirs]
        self.top_k = top_k

        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: Optional[bm25s.BM25] = None

        self._load_documents()
        self._build_bm25()

    # ------------------------------------------------------------------
    # Document loading — YAML description blocks only
    # ------------------------------------------------------------------

    def _load_documents(self) -> None:
        """
        Load description: prose blocks from all *.yaml files in docs_dirs.
        Billing codes come from the structured YAML field, not text extraction.
        """
        self._documents = []
        all_paths = []
        for d in self.docs_dirs:
            all_paths.extend(sorted(d.glob("*.yaml")))
        for path in all_paths:
            try:
                data: dict = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue

            description = (data.get("description") or "").strip()
            if not description:
                continue

            billing_codes = [
                c.upper() for c in (data.get("eligible_billing_codes") or [])
            ]

            doc = Document(
                page_content=description,
                metadata={
                    "source": path.name,
                    "full_path": str(path),
                    "billing_codes": billing_codes,
                },
            )
            self._documents.append(doc)

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Whitespace + punctuation tokenizer that preserves billing codes."""
        return re.findall(r"[A-Z0-9]+", text.upper())

    def _build_bm25(self) -> None:
        if not self._documents:
            return
        self._tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in self._documents
        ]
        self._bm25 = bm25s.BM25()
        self._bm25.index(self._tokenized_corpus)

    def _bm25_ranked_indices(self, query: str, boost_code: Optional[str] = None) -> list[int]:
        """
        Return document indices ranked by BM25 score.
        If boost_code is provided and a document's eligible_billing_codes contains
        that code, it is forced to rank first (exact-match priority).
        """
        if self._bm25 is None:
            return list(range(len(self._documents)))

        tokenized_query = self._tokenize(query)
        results, _ = self._bm25.retrieve([tokenized_query], k=len(self._documents))
        ranked = results[0].tolist()

        if boost_code:
            boosted = [
                i for i in ranked
                if boost_code.upper() in self._documents[i].metadata.get("billing_codes", [])
            ]
            others = [i for i in ranked if i not in set(boosted)]
            ranked = boosted + others

        return ranked

    # ------------------------------------------------------------------
    # Public retrieval interface
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[Document]:
        """
        BM25 search over policy description prose.
        Billing code exact matches are guaranteed to surface first.
        Returns up to self.top_k Documents with enriched metadata.
        """
        if not self._documents:
            return []

        exact_code = _query_contains_billing_code(query)
        ranked = self._bm25_ranked_indices(query, boost_code=exact_code)
        top_indices = ranked[: self.top_k]

        results: list[Document] = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "retrieval_rank": top_indices.index(idx) + 1,
                        "exact_code_match": (
                            exact_code is not None
                            and exact_code.upper() in doc.metadata.get("billing_codes", [])
                        ),
                    },
                )
            )

        return results

    def multi_query_search(self, policy_query: str, user_query: str) -> list[Document]:
        """
        Run two BM25 searches — one biased toward winning policies, one toward the
        user's question — then merge results deduplicated by source filename.
        Policy-biased results come first; user-query results fill remaining slots.
        Returns up to self.top_k Documents total.
        """
        if not self._documents:
            return []

        policy_docs = self.search(policy_query)
        user_docs = self.search(user_query)

        seen_sources: set[str] = set()
        merged: list[Document] = []

        for doc in policy_docs:
            src = doc.metadata.get("source", "")
            if src not in seen_sources:
                seen_sources.add(src)
                merged.append(doc)

        for doc in user_docs:
            src = doc.metadata.get("source", "")
            if src not in seen_sources and len(merged) < self.top_k:
                seen_sources.add(src)
                merged.append(doc)

        # Re-number retrieval ranks after merge
        for i, doc in enumerate(merged):
            doc.metadata["retrieval_rank"] = i + 1

        return merged

    def search_by_filenames(self, filenames: list[str]) -> list[Document]:
        """
        Return documents whose source filename is in the provided list.
        Used to retrieve prose for specific (e.g. ineligible) policies by name.
        """
        results: list[Document] = []
        for doc in self._documents:
            if doc.metadata.get("source", "") in filenames:
                results.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            **doc.metadata,
                            "retrieval_rank": len(results) + 1,
                            "exact_code_match": False,
                        },
                    )
                )
        return results
