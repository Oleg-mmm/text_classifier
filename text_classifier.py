import torch
import torch.nn as nn
from collections import Counter
from typing import Iterable, List


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


class Vocab:
    def __init__(self, min_freq: int = 2, max_size: int | None = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.token2id: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
        }
        self.id2token: dict[int, str] = {v: k for k, v in self.token2id.items()}

    def build(self, tokenized_texts: Iterable[List[str]]) -> "Vocab":
        counter: Counter[str] = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        most_common = counter.most_common()
        if self.max_size is not None:
            most_common = most_common[: self.max_size - len(self.token2id)]

        for token, freq in most_common:
            if freq < self.min_freq:
                continue
            if token in self.token2id:
                continue
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token

        return self

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token2id.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id2token.get(i, UNK_TOKEN) for i in ids]

    def __len__(self) -> int:
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        return token in self.token2id
    
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=4,
                 num_layers=2, max_len=128, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos(pos)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x.mean(dim=1)
        return self.fc(self.drop(x))