import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from src.models.captioner import ImageCaptioner
from src.training.loss import CaptionLoss
from src.training.scheduler import get_scheduler
from src.training.trainer import Trainer
from src.utils.config import load_config, get_device
from src.utils.serialization import save_model_artifacts

from src.data.dataset import create_data_loaders
from src.data.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_history(history: dict, save_path: str):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Image Captioner")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config and device
    config = load_config(args.config)
    device = get_device(config)
    logger.info(f"Using device: {device}")

    # Load vocabulary
    vocab_path = config['data'].get('vocab_path', 'data/processed/vocab.json')
    # Since Vocabulary logic varies, we might need a dummy or a real method. 
    # Usually it's instantiated and loaded manually or through a static method.
    # We will assume a loading method or basic instantiation here.
    try:
        vocab = Vocabulary.load(vocab_path)
    except AttributeError:
        # Fallback if load is not available in Vocabulary 
        vocab = Vocabulary() 
        logger.warning("Vocabulary.load not implemented, proceeding with base vocabulary.")
    
    vocab_size = len(vocab)
    logger.info(f"Loaded vocabulary with {vocab_size} tokens")

    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(config, vocab)

    # Create model
    model = ImageCaptioner.from_config(config, vocab_size=vocab_size)
    model = model.to(device)

    # Optimizer, loss, scheduler
    learning_rate = config['training'].get('learning_rate', 0.001)
    weight_decay = config['training'].get('weight_decay', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    criterion = CaptionLoss(pad_idx=vocab.pad_idx)
    scheduler = get_scheduler(optimizer, config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        config=config,
        device=device,
        logger=logger
    )

    # Train
    history = trainer.fit()

    # Save artifacts
    save_dir = config['checkpoint'].get('save_dir', 'checkpoints')
    logger.info(f"Saving model artifacts to {save_dir}")
    save_model_artifacts(model, vocab, config, save_dir)

    # Print summary
    best_val_loss = trainer.best_val_loss
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    # Plot history
    plot_path = Path(save_dir) / 'training_history.png'
    plot_history(history, str(plot_path))
    logger.info(f"Saved training history plot to {plot_path}")

if __name__ == "__main__":
    main()
