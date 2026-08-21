import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, List
from collections import Counter
import config


class Caltech101Subset(Dataset):
    """
    Wrapper dataset for Caltech101 that:
    a) Filters to only the top-N most populous classes
    b) Re-maps labels to 0..N-1
    c) Converts grayscale images to RGB
    d) Applies transforms
    """
    def __init__(self, root: str, top_n_classes: int, transform=None):
        self.base_dataset = datasets.Caltech101(root, download=True)
        self.transform = transform

        # Get all labels efficiently using the y attribute
        # Caltech101 stores labels as a list accessible via .y
        all_labels = self.base_dataset.y

        # Count class frequencies and find top-N
        class_counts = Counter(all_labels)
        top_classes = [c for c, _ in class_counts.most_common(top_n_classes)]
        top_classes_set = set(top_classes)

        # Build label mapping: old_label -> new_label (0..N-1)
        self.label_mapping = {old_label: new_label for new_label, old_label in enumerate(top_classes)}

        # Get class names
        self.class_names = [self.base_dataset.categories[c] for c in top_classes]

        # Filter indices to keep only top classes
        self.indices = [i for i, lbl in enumerate(all_labels) if lbl in top_classes_set]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]
        image, target = self.base_dataset[original_idx]

        # Convert grayscale to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transform
        if self.transform:
            image = self.transform(image)

        # Remap label
        new_target = self.label_mapping[target]
        return image, new_target


def get_caltech101_loaders(
    root: str,
    num_classes_subset: int,
    batch_size: int,
    num_workers: int,
    seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], int]:
    """
    Load Caltech101 dataset, subset to top classes, apply transforms, and split into loaders.

    Returns:
        train_loader, val_loader, test_loader, class_names, num_classes
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])

    subset_dataset = Caltech101Subset(root, num_classes_subset, transform=transform)
    class_names = subset_dataset.class_names
    num_classes = len(class_names)

    total_len = len(subset_dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        subset_dataset, [train_len, val_len, test_len], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names, num_classes
