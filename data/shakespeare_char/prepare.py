import os
import pickle
import requests
import numpy as np

# Download the tiny Shakespeare dataset
data_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(data_dir, 'input.txt')

if not os.path.exists(input_file_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    print("Downloading tiny Shakespeare dataset...")
    response = requests.get(url)
    response.raise_for_status()
    with open(input_file_path, 'w') as f:
        f.write(response.text)
    print(f"Downloaded {len(response.text):,} characters.")

with open(input_file_path, 'r') as f:
    data = f.read()

print(f"Dataset size: {len(data):,} characters")

# Build character-level vocabulary
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the entire dataset
encoded = np.array(encode(data), dtype=np.uint16)

# Train/val split (90% / 10%)
n = len(encoded)
train_data = encoded[:int(n * 0.9)]
val_data = encoded[int(n * 0.9):]

print(f"Train size: {len(train_data):,} tokens")
print(f"Val size:   {len(val_data):,} tokens")

# Save to binary files
train_data.tofile(os.path.join(data_dir, 'train.bin'))
val_data.tofile(os.path.join(data_dir, 'val.bin'))

# Save metadata (vocab mappings)
meta = {
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}
with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Done! Saved train.bin, val.bin, and meta.pkl")
