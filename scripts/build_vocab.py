"""
Build vocabulary from the Flickr8k captions CSV.
"""

import sys
import pandas as pd
from src.data import Vocabulary


def build_vocab():
    csv_file = 'data/captions.csv'
    output_file = 'data/vocab.pkl'
    min_freq = 5
    
    print(f"Building vocabulary from {csv_file}")
    print(f"Min frequency: {min_freq}")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} captions")
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=min_freq)
    
    for idx, row in df.iterrows():
        caption = row['caption']
        vocab.add_sentence(caption)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} captions...")
    
    print(f"Built vocabulary with {len(vocab.freqs)} unique tokens")
    vocab.build()
    print(f"Final vocabulary size (after min_freq={min_freq}): {len(vocab.word2idx)}")
    
    # Save
    vocab.save(output_file)
    print(f"\n✅ Vocabulary saved to {output_file}")


if __name__ == '__main__':
    build_vocab()
