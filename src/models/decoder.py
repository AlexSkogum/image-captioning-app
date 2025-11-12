import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Bahdanau-style additive attention over image features."""

    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attn_dim)  # encoder features
        self.decoder_att = nn.Linear(decoder_dim, attn_dim)  # decoder hidden
        self.full_att = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, num_pixels, encoder_dim)
        # decoder_hidden: (B, decoder_dim)
        att1 = self.encoder_att(encoder_out)  # (B, num_pixels, attn_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B,1,attn_dim)
        att = torch.tanh(att1 + att2)
        e = self.full_att(att).squeeze(2)  # (B, num_pixels)
        alpha = F.softmax(e, dim=1)  # (B, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=512, num_layers=1, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attn_dim=256)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden_state(self, encoder_out):
        mean_encoder = encoder_out.mean(dim=1)  # (B, encoder_dim)
        h = self.init_h(mean_encoder)
        c = self.init_c(mean_encoder)
        return h, c

    def forward(self, encoder_out, captions, lengths):
        # encoder_out: (B, num_pixels, encoder_dim)
        embeddings = self.embed(captions)  # (B, max_len, embed_dim)
        batch_size = encoder_out.size(0)
        max_len = max(lengths)
        device = embeddings.device
        
        # Initialize hidden states
        h, c = self.init_hidden_state(encoder_out)
        outputs = []
        alphas = []
        
        # Initialize a tensor to store outputs padded to max_len
        pred_outputs = torch.zeros(batch_size, max_len, self.fcn.out_features, device=device)
        
        for t in range(max_len):
            # Only process sequences that have not ended
            batch_size_t = sum([l > t for l in lengths])
            
            if batch_size_t == 0:
                break
            
            # Get attention encoding for active sequences
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            input_l = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h_t, c_t = self.decode_step(input_l, (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fcn(self.dropout(h_t))
            
            # Store outputs
            pred_outputs[:batch_size_t, t, :] = preds
            
            alphas.append(alpha)
            
            # Update hidden states for all sequences
            h_new = h.clone()
            c_new = c.clone()
            h_new[:batch_size_t] = h_t
            c_new[:batch_size_t] = c_t
            h = h_new
            c = c_new
        
        return pred_outputs, alphas

    def step(self, prev_word_idx, h, c, encoder_out):
        # one step for inference: prev_word_idx shape (B,)
        emb = self.embed(prev_word_idx)
        att_encoding, alpha = self.attention(encoder_out, h)
        input_l = torch.cat([emb, att_encoding], dim=1)
        h, c = self.decode_step(input_l, (h, c))
        scores = self.fcn(h)
        return scores, h, c, alpha
