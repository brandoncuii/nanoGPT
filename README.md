# nanoGPT

A minimal character-level GPT built from scratch in PyTorch. Trains on Shakespeare and generates new text in a similar style.

## Model Architecture

- **Decoder-only transformer** with causal self-attention
- 6 transformer blocks, 6 attention heads, 384 embedding dimensions
- ~10.8M parameters
- Character-level tokenization (65 unique characters)

## Project Structure

```
├── config.py                        # Model and training hyperparameters
├── model.py                         # GPT model (attention, MLP, transformer blocks)
├── train.py                         # Training loop with checkpointing
├── sample.py                        # Text generation from a trained checkpoint
└── data/shakespeare_char/
    └── prepare.py                   # Downloads and tokenizes the Shakespeare dataset
```

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install numpy requests torch
```

### 2. Prepare data

```bash
.venv/bin/python data/shakespeare_char/prepare.py
```

Downloads the tiny Shakespeare dataset (~1.1M characters), tokenizes it, and saves train/val splits.

### 3. Train

```bash
.venv/bin/python train.py
```

Trains for 3,000 iterations and saves the best checkpoint to `model.pt`. Training progress is logged every 500 steps.

### 4. Generate text

```bash
.venv/bin/python sample.py
```

Loads the checkpoint and generates Shakespeare-style text samples.

## Configuration

Edit `config.py` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layer` | 6 | Number of transformer blocks |
| `n_head` | 6 | Number of attention heads |
| `n_embd` | 384 | Embedding dimension |
| `block_size` | 256 | Max context length |
| `batch_size` | 16 | Sequences per batch |
| `max_iters` | 3000 | Training iterations |
| `learning_rate` | 1e-4 | AdamW learning rate |
| `dropout` | 0.2 | Dropout rate |

## Sample Output

```
KING Y VING ERY:
My dork not well I bideon is in we have with spelf in this
dow? Get you sed your in buth hers becences,
With comen think we so my be farther head?

CLARIO:
It a dobest unseer of Your in the gr
```