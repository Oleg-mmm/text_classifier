import torch
import torch.nn as nn


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