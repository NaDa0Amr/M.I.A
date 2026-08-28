import logging
from pathlib import Path
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, config, device, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    def train_epoch(self) -> float:
        """Run one training epoch. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for features, captions in tqdm(self.train_loader, desc='Training'):
            features = features.to(self.device)
            captions = captions.to(self.device)
            
            # Forward pass
            logits = self.model(features, captions)
            
            # Loss computation - targeting the entire captions 
            # (model predicts <start> from image, and rest from words)
            loss = self.criterion(logits, captions)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training'].get('grad_clip', 5.0))
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / max(1, num_batches)
    
    def validate_epoch(self) -> float:
        """Run validation. Returns average validation loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, captions in tqdm(self.val_loader, desc='Validation'):
                features = features.to(self.device)
                captions = captions.to(self.device)
                
                logits = self.model(features, captions)
                loss = self.criterion(logits, captions)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / max(1, num_batches)
    
    def fit(self) -> dict:
        """Full training loop with early stopping and checkpointing."""
        num_epochs = self.config['training'].get('num_epochs', 30)
        patience = self.config['training'].get('patience', 5)
        save_dir = self.config['checkpoint'].get('save_dir', 'checkpoints')
        
        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")
            
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            
            self.logger.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
            
            # LR scheduling
            self.scheduler.step(val_loss)
            
            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.config['checkpoint'].get('save_best', True):
                    self.save_checkpoint(save_dir, epoch, val_loss)
                    self.logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                self.logger.info(f"  No improvement. Patience: {self.patience_counter}/{patience}")
            
            # Early stopping
            if self.patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
        
        return self.history
    
    def save_checkpoint(self, save_dir: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, Path(save_dir) / 'best_model.pt')
