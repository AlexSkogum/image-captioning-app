import heapq
import math
import torch
import torch.nn.functional as F


def beam_search(decoder, encoder_out, start_idx, end_idx, beam_size=3, max_len=30, device='cpu'):
    """Simple beam search using decoder.step.

    decoder.step(prev_word_idx, h, c, encoder_out) -> scores, h, c, alpha
    encoder_out: (1, num_pixels, encoder_dim) or (B, num_pixels, encoder_dim) for batch size 1 expected
    Returns top sequences (list of idx lists) and scores (log-prob)
    """
    # only support single image at a time in this util
    assert encoder_out.size(0) == 1
    encoder_out = encoder_out.to(device)

    # init hidden with mean encoder
    h, c = decoder.init_hidden_state(encoder_out)

    # beam elements: (score, seq, h, c)
    seqs = [[start_idx]]
    scores = [0.0]
    states = [(h, c)]

    completed = []

    for _ in range(max_len):
        all_candidates = []
        for i, seq in enumerate(seqs):
            score = scores[i]
            h_i, c_i = states[i]
            prev = torch.tensor([seq[-1]], dtype=torch.long, device=device)
            with torch.no_grad():
                logits, h_new, c_new, _ = decoder.step(prev, h_i, c_i, encoder_out)
                logp = F.log_softmax(logits, dim=1).cpu().numpy().flatten()
            # consider top beam_size candidates locally
            topk_idx = logp.argsort()[-beam_size:][::-1]
            for idx in topk_idx:
                cand_seq = seq + [int(idx)]
                cand_score = score + float(logp[idx])
                all_candidates.append((cand_score, cand_seq, h_new, c_new))
        # select top beam_size
        ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        seqs = [c[1] for c in ordered]
        scores = [c[0] for c in ordered]
        states = [(c[2], c[3]) for c in ordered]
        # move completed
        new_seqs = []
        new_scores = []
        new_states = []
        for i, s in enumerate(seqs):
            if s[-1] == end_idx:
                completed.append((scores[i], s))
            else:
                new_seqs.append(s)
                new_scores.append(scores[i])
                new_states.append(states[i])
        seqs, scores, states = new_seqs, new_scores, new_states
        if not seqs:
            break

    if not completed:
        completed = list(zip(scores, seqs))
    completed = sorted(completed, key=lambda x: x[0], reverse=True)
    sequences = [c[1] for c in completed]
    scores = [c[0] for c in completed]
    return sequences, scores
