import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 vocab_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.decoder = nn.LSTM(input_size=hidden_size + vocab_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bias=True,
                               batch_first=False,
                               dropout=dropout,
                               bidirectional=False)
        self.to_h = nn.Sequential(nn.Linear(hidden_size,
                                            num_layers * hidden_size),
                                  nn.Tanh())
        self.init_step = nn.Embedding(1, vocab_size)
        self.prob = nn.Sequential(nn.Linear(hidden_size, vocab_size))

    def forward(self, x, conductors):
        # x of shape (track, seq_len, batch)
        tracks = []
        for i, conductor in enumerate(conductors):
            # conductor of shape (batch, hidden_size)
            batch_size = conductor.size(0)
            h = self.to_h(conductor).view(batch_size,
                                          self.num_layers,
                                          self.hidden_size).permute(1, 0, 2).contiguous()
            # (num_layers, batch, hidden_size)
            c = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=conductor.device)
            state = (h, c)
            init_step = self.init_step(torch.zeros(batch_size,
                                                   dtype=torch.long,
                                                   device=x.device)).unsqueeze(dim=0)  # (seq_len, batch, hidden_size)
            x_one_hot = torch.cat([init_step,
                                   F.one_hot(x[i, :-1],
                                             num_classes=self.vocab_size).float()],
                                  dim=0)  # (seq_len, batch, hidden_size + vocab_size)
            embeddings = torch.cat([x_one_hot,
                                    conductor.repeat(x_one_hot.size(0), 1, 1)],
                                   dim=-1)
            tracks.append(self.decoder(embeddings, state)[0])
        output = self.prob(torch.stack(tracks, dim=0))  # (num_tracks, seq_len, batch, vocab_size)
        return output.permute(2, 3, 0, 1)  # (batch, vocab_size, num_tracks, seq_len)

    def decode(self, conductors, length):
        tracks = []
        for conductor in conductors:
            device = conductor.device
            # conductor of shape (batch, hidden_size)
            batch_size = conductor.size(0)
            h = self.to_h(conductor).view(batch_size,
                                          self.num_layers,
                                          self.hidden_size).permute(1, 0, 2).contiguous()
            # (num_layers, batch, hidden_size)
            c = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device)
            state = (h, c)
            embedding = self.init_step(torch.zeros(batch_size,
                                                   dtype=torch.long,
                                                   device=device)).unsqueeze(dim=0)  # (batch, hidden_size)
            track = []
            for i in range(length):
                note = Categorical(F.softmax(embedding[-1], dim=-1)).sample()
                if i > 0:
                    track.append(note)
                note = F.one_hot(note, num_classes=self.vocab_size).float()
                note = torch.cat([note, conductor], dim=-1).unsqueeze(dim=0)
                embedding, state = self.decoder(note, state)
                embedding = self.prob(embedding)
            tracks.append(torch.stack(track, dim=-1))
        tracks = torch.stack(tracks, dim=1)
        return tracks  # (batch, num_tracks, seq_len)
