import argparse
import json
import os
import pickle
import re
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")


class Vocabulary:
    def __init__(self, min_freq: int = 5):
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.freqs = Counter()

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = TOKEN_PATTERN.sub(' ', text)
        tokens = [t for t in text.split() if t]
        return tokens

    def add_sentence(self, sent: str):
        for t in self.tokenize(sent):
            self.freqs[t] += 1

    def build(self):
        idx = max(self.idx2word.keys()) + 1
        for w, c in self.freqs.items():
            if c >= self.min_freq and w not in self.word2idx:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx += 1

    def encode(self, sent: str, max_len: int = 30) -> List[int]:
        tokens = self.tokenize(sent)
        idxs = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
        idxs = [self.word2idx['<start>']] + idxs + [self.word2idx['<end>']]
        if len(idxs) < max_len:
            idxs += [self.word2idx['<pad>']] * (max_len - len(idxs))
        return idxs[:max_len]

    def decode(self, idxs: List[int]) -> str:
        tokens = []
        for i in idxs:
            w = self.idx2word.get(i, '<unk>')
            if w in ['<start>', '<end>', '<pad>']:
                continue
            tokens.append(w)
        return ' '.join(tokens)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word, 'min_freq': self.min_freq}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        v = cls(min_freq=data.get('min_freq', 1))
        v.word2idx = data['word2idx']
        v.idx2word = data['idx2word']
        return v


class ImageCaptionDataset(Dataset):
    """Simple dataset: expects a CSV with columns `image_path,caption`."""

    def __init__(self, csv_file: str, images_dir: str, vocab: Vocabulary = None, transform=None, max_len: int = 30):
        df = pd.read_csv(csv_file)
        self.items = df[['image_path', 'caption']].values.tolist()
        self.images_dir = images_dir
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caption = self.items[idx]
        img = Image.open(os.path.join(self.images_dir, img_path)).convert('RGB')
        img = self.transform(img)
        if self.vocab:
            caption_idx = torch.tensor(self.vocab.encode(caption, max_len=self.max_len), dtype=torch.long)
        else:
            caption_idx = None
        return img, caption_idx, caption


def build_vocab_from_csv(captions_csv: str, out_path: str, min_freq: int = 5):
    df = pd.read_csv(captions_csv)
    vocab = Vocabulary(min_freq=min_freq)
    for c in df['caption'].astype(str).tolist():
        vocab.add_sentence(c)
    vocab.build()
    vocab.save(out_path)
    print(f"Saved vocab ({len(vocab.word2idx)}) to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-vocab', action='store_true')
    parser.add_argument('--captions-file', type=str)
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--min-freq', type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.build_vocab:
        assert args.captions_file, 'Provide --captions-file'
        out_path = os.path.join(args.out_dir, 'vocab.pkl')
        build_vocab_from_csv(args.captions_file, out_path, min_freq=args.min_freq)
