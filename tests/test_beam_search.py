import torch
from src.utils.beam_search import beam_search


class DummyDecoder:
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size

    def init_hidden_state(self, encoder_out):
        # return h,c
        B = encoder_out.size(0)
        return torch.zeros(B, 512), torch.zeros(B, 512)

    def step(self, prev_word_idx, h, c, encoder_out):
        # return logits, h, c, alpha
        B = encoder_out.size(0)
        logits = torch.log(torch.ones(B, self.vocab_size) / self.vocab_size)
        return logits, h, c, torch.ones(encoder_out.size(1))


def test_beam_search_runs():
    enc = torch.randn(1, 14 * 14, 512)
    dec = DummyDecoder(vocab_size=15)
    seqs, scores = beam_search(dec, enc, start_idx=1, end_idx=2, beam_size=2, max_len=5)
    assert isinstance(seqs, list)
    assert isinstance(scores, list)
