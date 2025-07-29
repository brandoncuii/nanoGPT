class GPTConfig:
    def __init__(self):
        #Model architecture parameters
        self.vocab_size = 65  # Size of the vocabulary (A-Z, a-z, 0-9, space, punctuation)
        self.block_size = 256  # Maximum context length
        self.n_layer = 6  # Number of transformer blocks
        self.n_head = 6  # Number of attention heads
        self.n_embd = 384  # Dimensionality of the embeddings

        # Training parameters
        self.learning_rate = 1e-4  # Learning rate for the optimizer
        self.device = "cpu" # Device to run the model on (cpu or cuda)
        self.batch_size = 64  # Number of sequences per batch
        self.max_iters = 5000  # Total number of training iterations
        self.dropout = 0.2
    
if __name__ == "__main__":
    config = GPTConfig()
    
    print("=== Our GPT Model Configuration ===")
    print(f"Vocabulary size: {config.vocab_size} characters")
    print(f"Context length: {config.block_size} characters") 
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of attention heads: {config.n_head}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")