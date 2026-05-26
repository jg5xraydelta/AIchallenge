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

if __name__ == "__main__":
    sample = (
        "Large language models work best when given focused context. "
        "When a document is too long to fit in the prompt, we split it "
        "into overlapping chunks.\n\n"
        "Each chunk should be small enough to embed, but large enough "
        "to carry meaning on its own. Overlap helps preserve context "
        "across boundaries so retrieved chunks remain coherent."
    ) * 5

    chunks = chunk_document(
        sample,
        chunk_size=300,
        chunk_overlap=50,
        metadata={"source": "demo.txt"},
    )

    for c in chunks:
        preview = c.text[:80].replace("\n", " ")
        print(f"[{c.index}] {c.start_char}-{c.end_char}  (~{c.token_estimate} tok)")
        print(preview + ("..." if len(c.text) > 80 else ""))
        print("-" * 60)
