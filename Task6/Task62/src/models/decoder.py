import torch
import torch.nn as nn

class CaptionDecoder(nn.Module):
    """
    LSTM-based caption decoder (Show and Tell architecture).
    
    During training:
        - Receives projected image features (batch, embed_dim) and captions (batch, seq_len)
        - Image feature is used as the first input at t=0
        - Teacher forcing: caption tokens are used as inputs for subsequent timesteps
        - Returns logits (batch, seq_len, vocab_size)
    
    During inference:
        - generate_step() is called autoregressively
    """
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int, num_layers: int = 1, dropout: float = 0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass.
        
        Args:
            features: projected image features (batch, embed_dim)
            captions: token IDs (batch, seq_len) including <start> and <end>
        
        Returns:
            logits: (batch, seq_len, vocab_size) - predictions for each timestep
                    At t=0, predicts from image feature
                    At t=1..n-1, predicts from caption tokens (teacher forcing)
        
        The targets should be captions[:, 1:] (shifted by 1)
        The logits output corresponds to predictions for captions[:, 1:]
        """
        # Don't use the last token as input (it's <end>, we don't predict after <end>)
        embeddings = self.dropout(self.embedding(captions[:, :-1]))  # (batch, seq_len-1, embed_dim)
        
        # Prepend image feature as first "word"
        img_embed = features.unsqueeze(1)  # (batch, 1, embed_dim)
        inputs = torch.cat([img_embed, embeddings], dim=1)  # (batch, seq_len, embed_dim)
        
        # LSTM forward
        hiddens, _ = self.lstm(inputs)  # (batch, seq_len, hidden_dim)
        logits = self.fc(self.dropout(hiddens))  # (batch, seq_len, vocab_size)
        
        return logits
    
    def generate_step(self, input_embed: torch.Tensor, hidden_state) -> tuple[torch.Tensor, tuple]:
        """
        Single step for autoregressive generation.
        
        Args:
            input_embed: (batch, 1, embed_dim) - either image feature or word embedding
            hidden_state: tuple of (h, c) or None for first step
        
        Returns:
            logits: (batch, vocab_size)
            hidden_state: updated (h, c) tuple
        """
        output, hidden_state = self.lstm(input_embed, hidden_state)
        logits = self.fc(output.squeeze(1))  # (batch, vocab_size)
        return logits, hidden_state
