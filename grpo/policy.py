import torch.nn as nn

class SmallPolicy(nn.Module):
    def __init__(self, vocab: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab, 128)
        self.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4), 2)
        self.lin = nn.Linear(128, vocab)

    def forward(self, idx):
        h = self.tr(self.emb(idx))
        return self.lin(h)