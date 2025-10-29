import base64
import io
import time

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

import torch

from src.infer import infer_image
from src.data import Vocabulary
from src.models.model import ImageCaptionModel


app = FastAPI()
MODEL = None
VOCAB = None


@app.on_event('startup')
async def load_model():
    global MODEL, VOCAB
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        VOCAB = Vocabulary.load('data/vocab.pkl')
    except Exception:
        VOCAB = None
    # load model if checkpoint exists
    try:
        ckpt = torch.load('checkpoints/best.pth', map_location=device)
        # must have vocab to know vocab size
        if VOCAB is None:
            raise RuntimeError('Vocab not found')
        # infer model dims from checkpoint parameters when possible
        state = ckpt.get('model_state', ckpt)
        try:
            enc_proj = state['encoder.conv_proj.weight']
            encoder_dim = enc_proj.shape[0]
        except Exception:
            encoder_dim = 512
        try:
            emb_w = state['decoder.embed.weight']
            embed_dim = emb_w.shape[1]
        except Exception:
            embed_dim = 512
        try:
            fcn_w = state['decoder.fcn.weight']
            hidden_dim = fcn_w.shape[1]
        except Exception:
            hidden_dim = 512
        vocab_size = ckpt.get('vocab_size', len(VOCAB.word2idx))
        MODEL = ImageCaptionModel(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, encoder_dim=encoder_dim)
        MODEL.load_state_dict(state)
        MODEL.to(device)
        MODEL.eval()
    except Exception as e:
        print('Warning: Could not load model at startup:', e)


@app.get('/health')
async def health():
    return {'status': 'ok', 'model_loaded': MODEL is not None}


@app.post('/caption')
async def caption(file: UploadFile = File(...), beam_size: int = Form(3), return_attention: bool = Form(False)):
    start = time.time()
    content = await file.read()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if MODEL is None or VOCAB is None:
        return JSONResponse({'error': 'Model or vocab not loaded'}, status_code=503)
    captions, scores, att_b64 = infer_image(MODEL, VOCAB, content, beam_size=beam_size, device=device, return_attentions=return_attention)
    duration = (time.time() - start) * 1000.0
    return {'captions': captions, 'scores': scores, 'duration_ms': duration, 'attention_base64': att_b64}
