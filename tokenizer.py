"""Byte-level BPE tokenizer, the same kind of tokenizer GPT-2 uses, trained on our own text."""
import json
import re
from collections import Counter

# GPT-2 first splits text into chunks (words with their leading space, numbers, punctuation, whitespace)
# so that merges never cross a word boundary. GPT-2's pattern uses \p{L} and \p{N}; Python's re module has
# no Unicode classes, so I use the ASCII version, which covers Tiny Shakespeare.
SPLIT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges or {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.cache = {}

    @property
    def vocab_size(self):
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size, verbose=False):
        """Start from the 256 byte values and repeatedly merge the most frequent adjacent pair into a new token."""
        words = Counter(tuple(chunk.encode("utf-8")) for chunk in SPLIT.findall(text))
        merges = {}
        for new_id in range(256, vocab_size):
            pairs = Counter()
            for ids, count in words.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += count
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            merges[best] = new_id
            words = {tuple(merge(list(ids), best, new_id)): count for ids, count in words.items()}
            if verbose and (new_id - 255) % 64 == 0:
                tok = cls(merges)
                print(f"  merge {new_id - 255:>4}: {tok.vocab[new_id]!r} (seen {pairs[best]} times)")
        return cls(merges)

    def encode_chunk(self, chunk):
        if chunk in self.cache:
            return self.cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # apply the earliest-learned merge first, exactly the order they were learned in
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self.cache[chunk] = ids
        return ids

    def encode(self, text):
        return [i for chunk in SPLIT.findall(text) for i in self.encode_chunk(chunk)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()]}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls({(a, b): new_id for a, b, new_id in json.load(f)["merges"]})


if __name__ == "__main__":
    import sys

    from data import load_tokenizer

    tok = load_tokenizer()
    text = sys.argv[1] if len(sys.argv) > 1 else "To be, or not to be: that is the question."
    ids = tok.encode(text)
    print(f"vocabulary: {tok.vocab_size} tokens (256 bytes + {len(tok.merges)} merges)")
    print(f"text:   {text!r} ({len(text)} characters)")
    print(f"ids:    {ids} ({len(ids)} tokens)")
    print(f"pieces: {' | '.join(repr(tok.decode([i]))[1:-1].replace(' ', '␣') for i in ids)}")
    print(f"decode(encode(text)) == text: {tok.decode(ids) == text}")
