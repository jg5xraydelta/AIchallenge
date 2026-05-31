"""
Document chunker for AI agent pipelines.

Splits long documents into overlapping chunks suitable for embedding,
retrieval, and LLM context windows. Uses recursive character splitting
with a hierarchy of separators so chunks respect natural boundaries
(paragraphs > sentences > words) when possible.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


@dataclass
class Chunk:
    text: str
    index: int                       # ordinal position in the document
    start_char: int                  # offset in the original document
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        # Rough heuristic: ~4 chars per token for English text.
        # Swap in tiktoken or your model's tokenizer for precision.
        return max(1, len(self.text) // 4)


class DocumentChunker:
    """Recursive character-based chunker with overlap."""

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator

    # ---- public API ----------------------------------------------------

    def split(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split a document into chunks, preserving char offsets."""
        pieces = self._recursive_split(text, self.separators)
        merged = self._merge(pieces)
        return self._to_chunks(merged, text, metadata or {})

    def split_many(self, docs: Iterable[Tuple[str, dict]]) -> List[Chunk]:
        """Split multiple (text, metadata) pairs into a single chunk list."""
        out: List[Chunk] = []
        for text, meta in docs:
            out.extend(self.split(text, meta))
        return out

    # ---- internals -----------------------------------------------------

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Split text down a hierarchy of separators until pieces fit chunk_size."""
        if len(text) <= self.chunk_size:
            return [text] if text else []

        # Pick the first separator that actually appears; "" = char-level fallback.
        sep = ""
        rest: List[str] = []
        for i, s in enumerate(separators):
            if s == "":
                break
            if s in text:
                sep = s
                rest = separators[i + 1:]
                break

        if sep == "":
            # Hard split on characters when no separator helps.
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size)]

        raw_parts = text.split(sep)
        if self.keep_separator:
            # Re-attach the separator to the end of each part except the last.
            parts = [p + sep for p in raw_parts[:-1]] + [raw_parts[-1]]
        else:
            parts = raw_parts

        out: List[str] = []
        for p in parts:
            if not p:
                continue
            if len(p) <= self.chunk_size:
                out.append(p)
            else:
                out.extend(self._recursive_split(p, rest))
        return out

    def _merge(self, pieces: List[str]) -> List[str]:
        """Greedily merge small pieces into chunks of ~chunk_size with overlap."""
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                chunks.append(current)
                tail = current[-self.chunk_overlap:] if self.chunk_overlap else ""
                current = tail + piece
        if current:
            chunks.append(current)
        return chunks

    def _to_chunks(
        self,
        chunk_texts: List[str],
        original: str,
        metadata: dict,
    ) -> List[Chunk]:
        """Attach char offsets and metadata. Uses search near a moving cursor."""
        out: List[Chunk] = []
        cursor = 0
        for i, t in enumerate(chunk_texts):
            start = original.find(t, max(0, cursor - self.chunk_overlap))
            if start == -1:
                start = cursor  # best-effort fallback
            end = start + len(t)
            cursor = end - self.chunk_overlap
            out.append(
                Chunk(
                    text=t,
                    index=i,
                    start_char=start,
                    end_char=end,
                    metadata={**metadata, "chunk_index": i},
                )
            )
        return out


# ---- convenience function ---------------------------------------------

def chunk_document(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    metadata: Optional[dict] = None,
    separators: Optional[List[str]] = None,
) -> List[Chunk]:
    """Chunk a document in one call. Returns a list of Chunk objects."""
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    chunks = chunker.split(text, metadata=metadata)
    return chunks


# ---- example usage ----------------------------------------------------

import argparse
import json
import os
from typing import Generator


def load_text_files(
    root: str, extensions: Optional[List[str]] = None
) -> Generator[Tuple[str, dict], None, None]:
    """Recursively yield (text, metadata) for files under `root`.

    Metadata contains `source`, `path`, and `filename`.
    """
    exts = set(extensions or [".txt"])
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() not in exts:
                continue
            path = os.path.join(dirpath, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                # best-effort: skip unreadable files
                continue
            meta = {
                "source": os.path.relpath(path, start=root),
                "path": path,
                "filename": fname,
            }
            yield text, meta


def chunk_text_files(
    root: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    extensions: Optional[List[str]] = None,
) -> List[Chunk]:
    """Load all text files under `root` and return a flat list of chunks."""
    docs = list(load_text_files(root, extensions=extensions))
    if not docs:
        return []
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_many(docs)


def save_chunks_jsonl(chunks: List[Chunk], outpath: str) -> None:
    """Serialize chunks to a JSONL file with useful metadata."""
    with open(outpath, "w", encoding="utf-8") as fh:
        for c in chunks:
            obj = {
                "text": c.text,
                "index": c.index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "token_estimate": c.token_estimate,
                "metadata": c.metadata,
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _cli():
    p = argparse.ArgumentParser(description="Chunk all text files under a folder")
    p.add_argument("root", help="Root folder to search for text files")
    p.add_argument("--out", help="Output JSONL file", default="chunks.jsonl")
    p.add_argument("--size", type=int, default=800, help="Chunk size")
    p.add_argument("--overlap", type=int, default=100, help="Chunk overlap")
    p.add_argument(
        "--ext", help="Comma-separated extensions (default: .txt)", default=".txt"
    )
    args = p.parse_args()
    exts = [e if e.startswith(".") else f".{e}" for e in args.ext.split(",")]
    chunks = chunk_text_files(
        args.root, chunk_size=args.size, chunk_overlap=args.overlap, extensions=exts
    )
    save_chunks_jsonl(chunks, args.out)
    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    _cli()
