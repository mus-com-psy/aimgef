import torch
import torch.nn as nn


class Conductor(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 num_tracks):
        super(Conductor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tracks = num_tracks
        self.conductor = nn.LSTM(input_size=hidden_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bias=True,
                                 batch_first=False,
                                 dropout=dropout,
                                 bidirectional=False)
        self.to_h = nn.Sequential(nn.Linear(hidden_size,
                                            num_layers * hidden_size),
                                  nn.Tanh())
        self.init_step = nn.Embedding(1, hidden_size)

    def forward(self, z):
        batch_size = z.size(0)
        h = self.to_h(z).view(batch_size,
                              self.num_layers,
                              self.hidden_size).permute(1, 0, 2).contiguous()  # (num_layers, batch, hidden_size)
        c = torch.zeros((self.num_layers, batch_size, self.hidden_size), device=z.device)
        state = (h, c)
        embedding = self.init_step(torch.zeros(batch_size,
                                               dtype=torch.long,
                                               device=z.device)).unsqueeze(dim=0)  # (seq_len, batch, hidden_size)
        output = []
        for i in range(self.num_tracks):
            embedding, state = self.conductor(embedding, state)
            output.append(embedding[-1])
        return output
