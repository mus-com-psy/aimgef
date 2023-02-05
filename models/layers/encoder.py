import torch
import torch.nn as nn


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        vocab_size
    ):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.track_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=True
        )
        self.measure_encoder = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=True
        )
        self.mu = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.log_var = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.sp = nn.Softplus()

    def forward(self, x):
        # x of shape (track, seq_len, batch)
        n_tracks = x.size(0)
        embeddings = self.embedding(x)  # (track, seq_len, batch, input_size)
        track_embedding = torch.stack(
            [self.track_encoder(embeddings[i])[0][-1]
             for i in range(n_tracks)],
            dim=0
        )  # (track, batch, num_directions * hidden_size)
        # (batch, num_directions * hidden_size * 2)
        measure_embedding = self.measure_encoder(track_embedding)[0][-1]
        mu = self.mu(measure_embedding)
        log_var = self.sp(self.log_var(measure_embedding))
        z = reparameterize(mu, log_var)  # (batch, hidden_size * 2)
        return z, mu, log_var
