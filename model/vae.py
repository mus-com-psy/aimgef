import torch.nn as nn

from model.layer.encoder import Encoder
from model.layer.decoder import Decoder
from model.layer.conductor import Conductor


class VAE(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 encoder_num_layers,
                 conductor_num_layers,
                 decoder_num_layers,
                 dropout,
                 vocab_size,
                 num_tracks):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size,
                               hidden_size,
                               encoder_num_layers,
                               dropout,
                               vocab_size)
        self.conductor = Conductor(hidden_size * 2,
                                   conductor_num_layers,
                                   dropout,
                                   num_tracks)
        self.decoder = Decoder(hidden_size * 2,
                               decoder_num_layers,
                               dropout,
                               vocab_size)

    def forward(self, x):
        # x of shape (batch, track, seq_len)
        x = x.permute(1, 2, 0)  # (track, seq_len, batch)
        z, mu, log_var = self.encoder(x)
        conductors = self.conductor(z)
        output = self.decoder(x, conductors)
        return output, mu, log_var

    def decode(self, z, length):
        conductors = self.conductor(z)
        output = self.decoder.decode(conductors, length)
        return output
