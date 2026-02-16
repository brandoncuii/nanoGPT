import os
import pickle
import torch
from torch.nn import functional as F
from model import GPT

# ---------- Configuration ----------
checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pt')
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'shakespeare_char')

num_samples = 3        # Number of text samples to generate
max_new_tokens = 500   # Length of each generated sample
temperature = 0.8      # 1.0 = no change, < 1.0 = more focused, > 1.0 = more random
top_k = None           # Set to an integer to limit sampling to top-k tokens

# ---------- Load checkpoint ----------
print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
config = checkpoint['config']
device = config.device

model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Loaded model from iteration {checkpoint['iter_num']} (val loss: {checkpoint['best_val_loss']:.4f})")

# ---------- Load vocab ----------
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# ---------- Generation ----------
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # Crop context to block_size if needed
        idx_cond = idx[:, -config.block_size:]
        logits = model(idx_cond)
        # Take logits at the last position
        logits = logits[:, -1, :] / temperature

        # Optionally apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# ---------- Sample ----------
# Start with a newline character as the seed
start = '\n'
start_ids = encode(start)
x = torch.tensor([start_ids], dtype=torch.long, device=device)

print(f"\nGenerating {num_samples} samples of {max_new_tokens} characters each...")
print(f"Temperature: {temperature}")
print("=" * 60)

for i in range(num_samples):
    y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)
    text = decode(y[0].tolist())
    print(f"\n--- Sample {i+1} ---\n")
    print(text)

print("\n" + "=" * 60)
