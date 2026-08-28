import torch.optim as optim

def get_scheduler(optimizer, config: dict):
    """Get learning rate scheduler based on config."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
