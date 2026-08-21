import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluate_clean(model, loader, device):
    """
    Evaluate model on clean test data.
    Returns: accuracy (float)
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total if total > 0 else 0.0

def evaluate_robust(model, loader, attack_fn, epsilon, criterion, device, **attack_kwargs):
    """
    Evaluate model on adversarially attacked test data.
    Returns: accuracy (float)
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, epsilon, criterion, device, **attack_kwargs)
        with torch.no_grad():
            outputs = model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total if total > 0 else 0.0

def evaluate_with_defense(model, loader, attack_fn, epsilon, criterion, device, defense_fn, **attack_kwargs):
    """
    Evaluate model with input_transform_defense applied at inference.
    For each batch: attack -> apply defense transform -> evaluate.
    Returns: accuracy (float)
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = attack_fn(model, images, labels, epsilon, criterion, device, **attack_kwargs)
        
        # Apply defense transform
        defended_images = defense_fn(adv_images)
        
        with torch.no_grad():
            outputs = model(defended_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return 100. * correct / total if total > 0 else 0.0

def evaluate_clean_with_defense(model, loader, device, defense_fn):
    """
    Clean accuracy WITH the input-transform defense active.

    This is the column that shows what the defense actually costs. Without it,
    any claim that the defense "only slightly degrades clean accuracy" is
    untested.
    """
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        defended = defense_fn(images)
        with torch.no_grad():
            outputs = model(defended)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total if total > 0 else 0.0


def evaluate_transfer(source_model, target_model, loader, attack_fn, epsilon,
                      criterion, device, **attack_kwargs):
    """
    Black-box transfer attack: craft perturbations against `source_model`,
    then measure how well `target_model` survives them.

    Accuracy well below the target's clean accuracy indicates the two models
    share a vulnerability surface; accuracy near clean means they do not.
    """
    source_model.eval()
    target_model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(source_model, images, labels, epsilon,
                               criterion, device, **attack_kwargs)
        with torch.no_grad():
            outputs = target_model(adv_images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total if total > 0 else 0.0


def run_transfer_matrix(models_dict, test_loader, attack_fn, epsilon, criterion,
                        device, **attack_kwargs):
    """Full source x target transfer matrix as a list of dicts."""
    rows = []
    names = list(models_dict.keys())
    for src in names:
        row = {'Source (attack crafted on)': src}
        for tgt in names:
            acc = evaluate_transfer(models_dict[src], models_dict[tgt],
                                    test_loader, attack_fn, epsilon,
                                    criterion, device, **attack_kwargs)
            row[tgt] = round(acc, 2)
        rows.append(row)
    return rows


def run_showdown(models_dict, test_loader, criterion, device, fgsm_eps, pgd_eps, pgd_alpha, pgd_steps):
    """
    Run the full showdown evaluation.
    models_dict = {
        'ResNet-34 Baseline': resnet_baseline,
        'ResNet-34 Hardened': resnet_hardened,
        'MobileNetV2 Baseline': mobilenet_baseline,
        'MobileNetV2 Hardened': mobilenet_hardened,
    }
    Evaluate each model on:
    - Clean accuracy
    - FGSM robust accuracy
    - PGD robust accuracy
    - FGSM + Input Transform defense accuracy
    
    Returns: results as list of dicts suitable for printing as table
    """
    from attacks import fgsm_attack, pgd_attack
    from defense import input_transform_defense
    
    results = []
    
    for name, model in models_dict.items():
        print(f"Evaluating {name}...")
        
        # Clean
        clean_acc = evaluate_clean(model, test_loader, device)

        # Clean WITH the input-transform defense (cost of the defense)
        clean_def_acc = evaluate_clean_with_defense(
            model, test_loader, device, input_transform_defense
        )
        
        # FGSM Robust
        fgsm_acc = evaluate_robust(model, test_loader, fgsm_attack, fgsm_eps, criterion, device)
        
        # PGD Robust
        pgd_acc = evaluate_robust(model, test_loader, pgd_attack, pgd_eps, criterion, device, alpha=pgd_alpha, num_steps=pgd_steps)
        
        # FGSM + Defense
        defended_acc = evaluate_with_defense(
            model, test_loader, fgsm_attack, fgsm_eps, criterion, device, input_transform_defense
        )
        
        results.append({
            'Model': name,
            'Clean Acc (%)': round(clean_acc, 2),
            'Clean + Defend Acc (%)': round(clean_def_acc, 2),
            'FGSM Robust Acc (%)': round(fgsm_acc, 2),
            'PGD Robust Acc (%)': round(pgd_acc, 2),
            'FGSM + Defend Acc (%)': round(defended_acc, 2)
        })
        
    return results

def print_results_table(results):
    """
    Pretty-print the showdown results as a formatted table.
    """
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("SHOWDOWN RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

def plot_results_chart(results, save_path):
    """
    Create a grouped bar chart comparing all models.
    X-axis: Model names
    Bars: Clean Acc, FGSM Robust Acc, PGD Robust Acc
    Use professional dark theme matching forensics.py style.
    Save to save_path.
    """
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dark theme settings
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
        
    models = df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.16
    
    metrics = ['Clean Acc (%)', 'Clean + Defend Acc (%)', 'FGSM Robust Acc (%)',
               'PGD Robust Acc (%)', 'FGSM + Defend Acc (%)']
    colors = ['#4CAF50', '#8BC34A', '#F44336', '#9C27B0', '#2196F3']
    
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, df[metric], width, label=metric, color=colors[i])
        
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=12)
    ax.set_title('Model Robustness Showdown', color='white', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, color='white', fontsize=10, rotation=15)
    
    legend = ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()

def save_results_csv(results, save_path):
    """
    Save results to CSV file.
    """
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
