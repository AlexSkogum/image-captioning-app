import io
import time
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.utils.beam_search import beam_search
from src.utils.visualize import to_base64, overlay_attention


def load_image_bytes(bts: bytes):
    img = Image.open(io.BytesIO(bts)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), img


def infer_image(model, vocab, image_bytes: bytes, beam_size:int=3, max_len:int=30, device='cpu', return_attentions=False):
    model.eval()
    img_tensor, pil_img = load_image_bytes(image_bytes)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        encoder_out, feat_map = model.encoder(img_tensor)

    if beam_size <= 1:
        # greedy
        seq = [vocab.word2idx['<start>']]
        h, c = model.decoder.init_hidden_state(encoder_out)
        alphas = []
        for _ in range(max_len):
            prev = torch.tensor([seq[-1]], device=device)
            scores, h, c, alpha = model.decoder.step(prev, h, c, encoder_out)
            idx = int(torch.argmax(scores, dim=1).item())
            seq.append(idx)
            alphas.append(alpha.cpu().numpy().flatten())
            if idx == vocab.word2idx['<end>']:
                break
        caption = vocab.decode(seq)
        att_img_b64 = None
        if return_attentions and alphas:
            att = np.mean(np.stack(alphas), axis=0)
            att_img = overlay_attention(pil_img, att, grid_size=(14,14))
            att_img_b64 = to_base64(att_img)
        return [caption], [0.0], att_img_b64
    else:
        sequences, scores = beam_search(model.decoder, encoder_out, start_idx=vocab.word2idx['<start>'], end_idx=vocab.word2idx['<end>'], beam_size=beam_size, max_len=max_len, device=device)
        captions = [vocab.decode(s) for s in sequences]
        # normalize scores by sequence length
        norm_scores = [s / max(1, len(seq)) for s, seq in zip(scores, sequences)]
        att_img_b64 = None
        return captions, norm_scores, att_img_b64
