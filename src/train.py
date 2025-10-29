import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import ImageCaptionDataset, Vocabulary
from src.models.model import ImageCaptionModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for imgs, captions, _ in dataloader:
        imgs = imgs.to(device)
        captions = captions.to(device)
        lengths = [ (captions[i]!=0).sum().item() for i in range(captions.size(0))]
        optimizer.zero_grad()
        outputs, _ = model(imgs, captions, lengths)
        # outputs: (B, max_len, vocab)
        targets = captions[:, 1:outputs.size(1)+1]
        loss = criterion(outputs.view(-1, outputs.size(2)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Coerce numeric types from config to avoid string parsing issues
    try:
        cfg['training']['lr'] = float(cfg['training']['lr'])
    except Exception:
        pass
    try:
        cfg['training']['batch_size'] = int(cfg['training']['batch_size'])
    except Exception:
        pass
    try:
        cfg['training']['epochs'] = int(cfg['training']['epochs'])
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() and not args.dev else 'cpu')

    vocab = Vocabulary.load('data/vocab.pkl')
    dataset = ImageCaptionDataset(cfg['dataset']['captions_csv'], cfg['dataset']['images_dir'], vocab=vocab, max_len=cfg['inference']['max_len'])
    dataloader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)

    model = ImageCaptionModel(vocab_size=len(vocab.word2idx), embed_dim=cfg['model']['embedding_dim'], hidden_dim=cfg['model']['hidden_dim'], encoder_dim=cfg['model']['embedding_dim'], num_layers=cfg['model']['num_layers'], dropout=cfg['model']['dropout'], fine_tune_encoder=cfg['model']['fine_tune_encoder'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx['<pad>'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg['training']['lr'])

    writer = SummaryWriter(log_dir='runs/exp')

    best_loss = float('inf')
    for epoch in range(1, (2 if args.dev else cfg['training']['epochs']) + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        writer.add_scalar('train/loss', avg_loss, epoch)
        print(f'Epoch {epoch} loss={avg_loss:.4f} time={(time.time()-t0):.1f}s')
        # checkpoint
        ckpt = {'model_state': model.state_dict(), 'vocab_size': len(vocab.word2idx)}
        torch.save(ckpt, f'checkpoints/epoch_{epoch}.pth')
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(ckpt, 'checkpoints/best.pth')

    writer.close()


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    main()
