import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_size: int,
        padding_idx: int,
        embed_size=128,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
    ):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        hidden = hidden[-1]
        out = self.dropout(hidden)
        out = self.fc(out)
        return out
