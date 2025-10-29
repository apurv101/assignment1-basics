# bpe_tokenizer.py
from __future__ import annotations
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple

import regex as re  # pip install regex

# GPT-2 style regex (requires 'regex' package with \p classes and negative lookahead)
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# ----------------------------
# Utilities
# ----------------------------
def _byte(b: int) -> bytes:
    """Single-byte helper."""
    return bytes([b])

def _concat(b1: bytes, b2: bytes) -> bytes:
    """Concatenate byte tokens."""
    return b1 + b2

def _escape_re(s: str) -> str:
    return re.escape(s)

def _split_on_specials(text: str, special_tokens: List[str]) -> List[Tuple[str, bool]]:
    """
    Split text into (piece, is_special) where specials are preserved as whole pieces.
    """
    if not special_tokens:
        return [(text, False)]
    # Build one big alternation for specials; escape everything
    specials_alt = "|".join(_escape_re(tok) for tok in sorted(special_tokens, key=len, reverse=True))
    # Capture specials to keep them in output
    pattern = re.compile(f"({specials_alt})")
    parts = pattern.split(text)
    out: List[Tuple[str, bool]] = []
    for part in parts:
        if part == "":
            continue
        if part in special_tokens:
            out.append((part, True))
        else:
            out.append((part, False))
    return out

def _pretokenize(text: str) -> List[str]:
    """GPT-2 style pretokenization into coarse tokens."""
    return [m.group(0) for m in re.finditer(GPT2_PAT, text)]

def _utf8_bytes(s: str) -> List[bytes]:
    """
    Represent a string as a list of single-byte tokens (bytes objects of length 1).
    """
    b = s.encode("utf-8")
    return [_byte(x) for x in b]

def _apply_full_merge_pass(seq: List[bytes], a: bytes, b: bytes) -> List[bytes]:
    """Replace all occurrences of adjacent pair (a,b) with a+b in a single pass."""
    if not seq:
        return seq
    merged: List[bytes] = []
    i = 0
    ab = _concat(a, b)
    n = len(seq)
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            merged.append(ab)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return merged

# ----------------------------
# BPE Training
# ----------------------------
@dataclass(frozen=True)
class BPETokenizerArtifacts:
    vocab: Dict[int, bytes]                          # id -> bytes
    merges: List[Tuple[bytes, bytes]]                # list of (a,b), order of creation
    special_tokens: List[str]

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str] | None = None,
) -> BPETokenizerArtifacts:
    """
    Train a byte-level BPE tokenizer from a text file.
    - Starts with 256 byte tokens + given special tokens.
    - Uses GPT-2 regex pretokenization.
    - Never merges across special-token boundaries.
    - Breaks merge ties lexicographically (max by (count, pair)).
    Returns:
        BPETokenizerArtifacts(vocab, merges, special_tokens)
    """
    special_tokens = list(special_tokens or [])

    # 1) Build the initial vocabulary (0..255 are single-byte tokens)
    vocab: Dict[int, bytes] = {i: _byte(i) for i in range(256)}
    next_id = 256

    # Reserve IDs for special tokens (appended in given order)
    for tok in special_tokens:
        tok_bytes = tok.encode("utf-8")
        vocab[next_id] = tok_bytes
        next_id += 1

    # 2) Pre-tokenize and build frequency table of byte sequences (as tuples of bytes objects)
    # We do not allow merges that cross special tokens, so split first.
    sequences: Counter[Tuple[bytes, ...]] = Counter()

    def add_text_block(block: str):
        # Pre-tokenize block with GPT-2 regex; each pretok -> UTF-8 bytes -> sequence of byte tokens
        for pretok in _pretokenize(block):
            seq = tuple(_utf8_bytes(pretok))
            if seq:  # skip empty
                sequences[seq] += 1

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            for piece, is_special in _split_on_specials(raw_line.rstrip("\n"), special_tokens):
                if is_special:
                    # Special tokens do not participate in merges; they are their own whole unit.
                    # We simply don't add them to the sequences multiset for counting merges.
                    continue
                if piece:
                    add_text_block(piece)

    # Helper to compute pair frequencies across all sequences
    def compute_pair_counts(seq_counter: Counter[Tuple[bytes, ...]]) -> Counter[Tuple[bytes, bytes]]:
        pair_counts: Counter[Tuple[bytes, bytes]] = Counter()
        for seq, freq in seq_counter.items():
            if len(seq) < 2:
                continue
            # Count adjacent pairs (with multiplicity) in this sequence
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq
        return pair_counts

    merges: List[Tuple[bytes, bytes]] = []

    # The current vocab size includes bytes + specials + newly created merges
    def current_vocab_size() -> int:
        # + merges (each merge creates a new token)
        return 256 + len(special_tokens) + len(merges)

    # Main merge loop
    while current_vocab_size() < vocab_size:
        pair_counts = compute_pair_counts(sequences)
        if not pair_counts:
            break  # nothing left to merge

        # Pick the most frequent pair; tie-break lexicographically on the pair itself
        # max by (count, pair)
        best_pair, best_count = None, None
        for pair, c in pair_counts.items():
            if best_pair is None or (c, pair) > (best_count, best_pair):
                best_pair, best_count = pair, c

        if best_pair is None or best_count is None or best_count == 0:
            break

        a, b = best_pair
        new_token = _concat(a, b)

        # Update sequences by replacing all (a,b) with new_token
        new_sequences: Counter[Tuple[bytes, ...]] = Counter()
        for seq, freq in sequences.items():
            if len(seq) < 2:
                new_sequences[seq] += freq
                continue
            # Fast path: if the pair never occurs, keep as is
            has_pair = False
            for i in range(len(seq) - 1):
                if seq[i] == a and seq[i + 1] == b:
                    has_pair = True
                    break
            if not has_pair:
                new_sequences[seq] += freq
                continue

            # Replace occurrences in a single pass
            replaced = _apply_full_merge_pass(list(seq), a, b)
            new_sequences[tuple(replaced)] += freq

        sequences = new_sequences
        merges.append((a, b))

        # Add the new token to vocab (ID assignment follows merge creation order)
        vocab[next_id] = new_token
        next_id += 1

    return BPETokenizerArtifacts(vocab=vocab, merges=merges, special_tokens=special_tokens)

# ----------------------------
# Tokenizer (encode/decode)
# ----------------------------
class Tokenizer:
    """
    Byte-level BPE Tokenizer.
    - Applies GPT-2 pretokenization within non-special segments.
    - Applies merges in order of creation (full-pass replacement each step).
    - Special tokens are preserved atomically.
    """
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ):
        self.vocab_id2bytes: Dict[int, bytes] = dict(vocab)
        self.vocab_bytes2id: Dict[bytes, int] = {v: k for k, v in self.vocab_id2bytes.items()}
        self.merges: List[Tuple[bytes, bytes]] = list(merges)
        self.special_tokens: List[str] = list(special_tokens or [])

        # Pre-encode special tokens as bytes for quick compare
        self._special_bytes: Dict[bytes, int] = {}
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            tok_id = self.vocab_bytes2id.get(b)
            if tok_id is None:
                # If provided but not in vocab, append (rare path)
                tok_id = max(self.vocab_id2bytes) + 1
                self.vocab_id2bytes[tok_id] = b
                self.vocab_bytes2id[b] = tok_id
            self._special_bytes[b] = tok_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Expect { "ids": [int...], "bytes_hex": ["..."] } OR {"vocab": {id:hex}}
        if "vocab" in raw:
            id2bytes = {int(k): bytes.fromhex(v) for k, v in raw["vocab"].items()}
        else:
            ids = raw["ids"]
            hexes = raw["bytes_hex"]
            id2bytes = {int(i): bytes.fromhex(h) for i, h in zip(ids, hexes)}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            # stored as [["hex_a","hex_b"], ...]
            raw_merges = json.load(f)
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges]
        return cls(id2bytes, merges, special_tokens=special_tokens)

    # ---- Serialization helpers (optional but handy) ----
    def save(self, vocab_filepath: str, merges_filepath: str) -> None:
        # Save deterministic order by ID
        items = sorted(self.vocab_id2bytes.items())
        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ids": [i for i, _ in items],
                    "bytes_hex": [b.hex() for _, b in items],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(merges_filepath, "w", encoding="utf-8") as f:
            json.dump([[a.hex(), b.hex()] for (a, b) in self.merges], f, ensure_ascii=False, indent=2)

    # ---- Encoding / Decoding ----
    def _encode_pretoken(self, s: str) -> List[int]:
        # Start as single-byte tokens
        seq: List[bytes] = _utf8_bytes(s)
        # Apply merges in order
        for (a, b) in self.merges:
            seq = _apply_full_merge_pass(seq, a, b)
        # Map to IDs (must exist)
        out: List[int] = []
        for token in seq:
            tid = self.vocab_bytes2id.get(token)
            if tid is None:
                # Should not happen if training and encoding are consistent, but fallback:
                # insert unknown as raw bytes (byte-by-byte)
                for bb in token:
                    out.append(self.vocab_bytes2id[_byte(bb)])
            else:
                out.append(tid)
        return out

    def encode(self, text: str) -> List[int]:
        """
        Encode a string into token IDs, preserving special tokens intact.
        """
        ids: List[int] = []
        for piece, is_special in _split_on_specials(text, self.special_tokens):
            if is_special:
                tok_bytes = piece.encode("utf-8")
                ids.append(self.vocab_bytes2id[tok_bytes])  # guaranteed present
            else:
                # pretokenize and encode each pretoken independently (no merges across)
                for pretok in _pretokenize(piece):
                    ids.extend(self._encode_pretoken(pretok))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (useful for large files).
        """
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back into text (UTF-8), replacing malformed sequences if any.
        """
        buf = bytearray()
        for tid in ids:
            b = self.vocab_id2bytes.get(int(tid))
            if b is None:
                continue
            buf.extend(b)
        return bytes(buf).decode("utf-8", errors="replace")


# ----------------------------
# Example CLI usage (optional)
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Train/Use a byte-level BPE tokenizer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--input", required=True, help="Path to UTF-8 text file")
    p_train.add_argument("--vocab_size", type=int, required=True)
    p_train.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    p_train.add_argument("--out_dir", required=True)

    p_tok = sub.add_parser("encode")
    p_tok.add_argument("--vocab", required=True)
    p_tok.add_argument("--merges", required=True)
    p_tok.add_argument("--text", required=True)
    p_tok.add_argument("--special", nargs="*", default=["<|endoftext|>"])

    args = parser.parse_args()

    if args.cmd == "train":
        arts = train_bpe(args.input, args.vocab_size, special_tokens=args.special)
        os.makedirs(args.out_dir, exist_ok=True)
        tok = Tokenizer(arts.vocab, arts.merges, special_tokens=arts.special_tokens)
        tok.save(os.path.join(args.out_dir, "vocab.json"),
                 os.path.join(args.out_dir, "merges.json"))
        print(f"Trained. Vocab size: {len(arts.vocab)} (+{len(arts.merges)} merges). Saved to {args.out_dir}")

    elif args.cmd == "encode":
        tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=args.special)
        print(tok.encode(args.text))
