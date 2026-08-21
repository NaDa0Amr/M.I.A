import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import copy
from config import IMAGENET_MEAN, IMAGENET_STD

def adversarial_train_one_epoch(model, loader, optimizer, criterion, attack_fn, epsilon, device, mix_ratio=0.5):
  
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        batch_size = images.size(0)
        split_idx = int(batch_size * (1 - mix_ratio))
        
        clean_images = images[:split_idx]
        clean_labels = labels[:split_idx]
        
        adv_images_src = images[split_idx:]
        adv_labels = labels[split_idx:]
        
        if len(adv_images_src) > 0:
            # We need to compute gradients with respect to the input for the attack
            # Model needs to be temporarily in eval mode or have gradients enabled correctly
            model.eval()
            adv_images = attack_fn(model, adv_images_src, adv_labels, epsilon, criterion, device)
            model.train()
            
            mixed_images = torch.cat([clean_images, adv_images], dim=0)
            mixed_labels = torch.cat([clean_labels, adv_labels], dim=0)
        else:
            mixed_images = clean_images
            mixed_labels = clean_labels
            
        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = criterion(outputs, mixed_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * mixed_images.size(0)
        _, predicted = outputs.max(1)
        total += mixed_labels.size(0)
        correct += predicted.eq(mixed_labels).sum().item()
        
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def adversarial_train_model(model, train_loader, val_loader, epochs, lr, attack_fn, epsilon, device, model_name, save_dir):

    import os
    from train import validate
    
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = adversarial_train_one_epoch(
            model, train_loader, optimizer, criterion, attack_fn, epsilon, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(save_dir, f"{model_name}_best.pth"))
            
    if os.path.exists(os.path.join(save_dir, f"{model_name}_best.pth")):
        from utils import load_model
        model = load_model(model, os.path.join(save_dir, f"{model_name}_best.pth"))
        
    return model, history

def input_transform_defense(images, image_size=224, jpeg_quality=75):

    import torchvision.transforms as T
    from utils import denormalize
    
    device = images.device
    batch_size = images.size(0)
    
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    
    # Denormalize
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    transformed_images = []
    
    for i in range(batch_size):
        img_tensor = images_denorm[i]
        
        # Convert to PIL
        img_pil = T.ToPILImage()(img_tensor.cpu())
        
        # 1. Random resize
        scale_factor = torch.empty(1).uniform_(0.9, 1.1).item()
        new_size = int(image_size * scale_factor)
        img_resized = T.Resize((new_size, new_size))(img_pil)
        
        # 2. Re-crop/resize back to image_size
        if new_size > image_size:
            img_cropped = T.CenterCrop(image_size)(img_resized)
        else:
            img_cropped = T.Resize((image_size, image_size))(img_resized)
            
        # 3. JPEG compression simulation
        buffer = io.BytesIO()
        img_cropped.save(buffer, format='JPEG', quality=jpeg_quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer)
        
        # Convert back to tensor
        img_transformed = T.ToTensor()(img_jpeg).to(device)
        transformed_images.append(img_transformed)
        
    transformed_batch = torch.stack(transformed_images)
    
    # Re-normalize
    transformed_batch = (transformed_batch - mean) / std
    return transformed_batch
