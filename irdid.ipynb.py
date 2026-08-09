"""
BLOCK: Vocabulary / Tokenizer
------------------------------
Manual word-level tokenizer + vocab builder. No HuggingFace tokenizers,
no spaCy, no nltk 'parse' utilities -- just regex + Counter.
"""

import re
from collections import Counter


class Vocab:

    def __init__(self, captions, min_freq=2):
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        counter = Counter()
        for cap in captions:
            counter.update(self.tokenize(cap))

        specials = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        words = [w for w, c in counter.items() if c >= min_freq]

        self.itos = specials + sorted(words)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

        self.pad_id = self.stoi[self.pad_token]
        self.sos_id = self.stoi[self.sos_token]
        self.eos_id = self.stoi[self.eos_token]
        self.unk_id = self.stoi[self.unk_token]

    @staticmethod
    def tokenize(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)   # strip punctuation manually
        text = re.sub(r"\s+", " ", text).strip()
        return text.split(" ") if text else []

    def encode(self, text, max_len=40):
        tokens = self.tokenize(text)
        ids = [self.sos_id]
        for t in tokens[: max_len - 2]:
            ids.append(self.stoi.get(t, self.unk_id))
        ids.append(self.eos_id)
        while len(ids) < max_len:
            ids.append(self.pad_id)
        return ids[:max_len]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.sos_id, self.pad_id):
                continue
            words.append(self.itos[i])
        return " ".join(words)

    def __len__(self):
        return len(self.itos)
