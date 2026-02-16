import os
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from model import GPT
from config import GPTConfig

# ---------- Load data ----------
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'shakespeare_char')

# Load vocabulary metadata
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

# Load train and val data as memory-mapped arrays
train_data = np.fromfile(os.path.join(data_dir, 'train.bin'), dtype=np.uint16)
val_data = np.fromfile(os.path.join(data_dir, 'val.bin'), dtype=np.uint16)

print(f"Loaded {len(train_data):,} train tokens, {len(val_data):,} val tokens")
print(f"Vocab size: {meta['vocab_size']}")

# ---------- Config ----------
config = GPTConfig()
config.vocab_size = meta['vocab_size']

device = config.device
batch_size = config.batch_size
block_size = config.block_size
max_iters = config.max_iters
learning_rate = config.learning_rate
eval_interval = 500
eval_iters = 50

# ---------- Helpers ----------
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------- Model ----------
model = GPT(config)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model has {total_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ---------- Training loop ----------
print(f"\nTraining for {max_iters} iterations...")
print(f"Device: {device} | Batch size: {batch_size} | Block size: {block_size}\n")

best_val_loss = float('inf')
out_dir = os.path.dirname(os.path.abspath(__file__))

for iter_num in range(max_iters):

    # Evaluate periodically
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter_num:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}")

        # Save checkpoint if val loss improved
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'model.pt'))
            print(f"  -> Saved checkpoint (val loss: {best_val_loss:.4f})")

    # Forward pass
    xb, yb = get_batch('train')
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to {os.path.join(out_dir, 'model.pt')}")
