import torch
import torch.nn as nn

from src.models.encoder import EncoderCNN
from src.models.decoder import DecoderRNN


class ImageCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, encoder_dim=512, num_layers=1, dropout=0.5, fine_tune_encoder=False):
        super().__init__()
        self.encoder = EncoderCNN(embed_dim=encoder_dim, fine_tune=fine_tune_encoder)
        self.decoder = DecoderRNN(embed_dim=embed_dim, decoder_dim=hidden_dim, vocab_size=vocab_size, encoder_dim=encoder_dim, num_layers=num_layers, dropout=dropout)

    def forward(self, images, captions, lengths):
        encoder_out, _ = self.encoder(images)
        outputs, alphas = self.decoder(encoder_out, captions, lengths)
        return outputs, alphas
