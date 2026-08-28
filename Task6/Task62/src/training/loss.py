import torch
import torch.nn as nn

class CaptionLoss(nn.Module):
    """Cross-entropy loss that ignores padding tokens."""
    def __init__(self, pad_idx: int):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, vocab_size)
            targets: (batch, seq_len)
        """
        # Reshape for CrossEntropyLoss: (batch*seq_len, vocab_size) and (batch*seq_len,)
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)
        return self.criterion(logits, targets)
