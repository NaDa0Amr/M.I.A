"""
Adversarial ML Security Investigation -- Main Orchestrator
=========================================================
Runs all 5 phases of the investigation:
1. Data loading & model fine-tuning
2. Adversarial attack implementation & epsilon sweep
3. Explainability analysis
4. Forensic visualization
5. Model hardening & "The Showdown" evaluation

Usage: python main.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config import *
from utils import set_seed, save_model, load_model
from data_pipeline import get_caltech101_loaders
from models import build_resnet34, build_mobilenetv2
from train import train_model
from attacks import fgsm_attack, pgd_attack, epsilon_sweep, evaluate_under_attack
from explainability import compute_saliency, compute_gradcam, overlay_heatmap
from forensics import (plot_forensic_panel, compare_models_forensic,
                       compare_two_models_forensic, find_attack_successes)
from defense import adversarial_train_model, input_transform_defense
from evaluation import (run_showdown, print_results_table, plot_results_chart,
                        save_results_csv, run_transfer_matrix)

def plot_training_history(history, model_name, save_dir):
    """Plot training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title(f'{model_name} Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_title(f'{model_name} Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=150)
    plt.close()

def plot_epsilon_sweep(resnet_sweep, mobilenet_sweep, attack_name, save_dir):
    """Plot epsilon sweep accuracy curves for both models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    epsilons = list(resnet_sweep.keys())
    resnet_accs = [v * 100 for v in resnet_sweep.values()]
    mobilenet_accs = [v * 100 for v in mobilenet_sweep.values()]

    ax.plot(epsilons, resnet_accs, marker='o', linewidth=2, label='ResNet-34', color='#2196F3')
    ax.plot(epsilons, mobilenet_accs, marker='s', linewidth=2, label='MobileNetV2', color='#F44336')

    ax.set_title(f'{attack_name} Epsilon Sweep', fontsize=14)
    ax.set_xlabel('Epsilon', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{attack_name}_epsilon_sweep.png'), dpi=150)
    plt.close()

def main():
    print('=' * 70)
    print('  ADVERSARIAL ML SECURITY INVESTIGATION')
    print('  Production-Grade End-to-End Analysis')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    # ===========================================================
    # PHASE 1: Data Loading & Model Fine-Tuning
    # ===========================================================
    phase1_start = time.time()
    print('\n' + '=' * 50)
    print('  PHASE 1: Data Loading & Model Fine-Tuning')
    print('=' * 50)

    try:
        train_loader, val_loader, test_loader, class_names, num_classes = get_caltech101_loaders(
            root=DATA_ROOT, num_classes_subset=NUM_CLASSES_SUBSET,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED
        )
        print(f'Classes: {num_classes}, Class names: {class_names}')
        print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')

        # Train ResNet-34
        print('\n--- Training ResNet-34 ---')
        resnet = build_resnet34(num_classes, DEVICE)
        resnet, resnet_history = train_model(resnet, train_loader, val_loader, CLEAN_EPOCHS, LR, DEVICE, 'ResNet34', OUTPUT_DIR)
        plot_training_history(resnet_history, 'ResNet34', OUTPUT_DIR)

        # Train MobileNetV2
        print('\n--- Training MobileNetV2 ---')
        mobilenet = build_mobilenetv2(num_classes, DEVICE)
        mobilenet, mobilenet_history = train_model(mobilenet, train_loader, val_loader, CLEAN_EPOCHS, LR, DEVICE, 'MobileNetV2', OUTPUT_DIR)
        plot_training_history(mobilenet_history, 'MobileNetV2', OUTPUT_DIR)
    except Exception as e:
        print(f"Error in Phase 1: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"Phase 1 completed in {time.time() - phase1_start:.2f} seconds.")

    # ===========================================================
    # PHASE 2: Adversarial Attack & Epsilon Sweep
    # ===========================================================
    phase2_start = time.time()
    print('\n' + '=' * 50)
    print('  PHASE 2: Adversarial Attacks & Calibration')
    print('=' * 50)

    try:
        # FGSM Epsilon Sweep on both models
        print('\nFGSM Epsilon Sweep - ResNet-34:')
        resnet_fgsm_sweep = epsilon_sweep(resnet, test_loader, fgsm_attack, EPSILON_SWEEP, criterion, DEVICE)
        print('FGSM Epsilon Sweep - MobileNetV2:')
        mobilenet_fgsm_sweep = epsilon_sweep(mobilenet, test_loader, fgsm_attack, EPSILON_SWEEP, criterion, DEVICE)
        plot_epsilon_sweep(resnet_fgsm_sweep, mobilenet_fgsm_sweep, 'FGSM', OUTPUT_DIR)

        # PGD Epsilon Sweep on both models
        print('\nPGD Epsilon Sweep - ResNet-34:')
        resnet_pgd_sweep = epsilon_sweep(resnet, test_loader, pgd_attack, EPSILON_SWEEP, criterion, DEVICE, alpha=PGD_ALPHA, num_steps=PGD_STEPS)
        print('PGD Epsilon Sweep - MobileNetV2:')
        mobilenet_pgd_sweep = epsilon_sweep(mobilenet, test_loader, pgd_attack, EPSILON_SWEEP, criterion, DEVICE, alpha=PGD_ALPHA, num_steps=PGD_STEPS)
        plot_epsilon_sweep(resnet_pgd_sweep, mobilenet_pgd_sweep, 'PGD', OUTPUT_DIR)
    except Exception as e:
        print(f"Error in Phase 2: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"Phase 2 completed in {time.time() - phase2_start:.2f} seconds.")

    # ===========================================================
    # PHASE 3 & 4: Explainability & Forensics
    # ===========================================================
    phase34_start = time.time()
    print('\n' + '=' * 50)
    print('  PHASE 3 & 4: Explainability & Forensic Analysis')
    print('=' * 50)

    try:
        resnet_target_layer = resnet.layer4[-1]        # Last conv block for ResNet
        mobilenet_target_layer = mobilenet.features[-1]  # Last conv block for MobileNet

        # Collect candidate samples across several batches, then keep only the
        # ones where the attack ACTUALLY flips the prediction. Selecting on the
        # clean prediction alone produces panels that document a failed attack.
        pool_images, pool_labels = [], []
        for bi, (bimg, blbl) in enumerate(test_loader):
            pool_images.append(bimg)
            pool_labels.append(blbl)
            if bi >= 2:
                break
        pool_images = torch.cat(pool_images)
        pool_labels = torch.cat(pool_labels)

        print('  Searching for samples where FGSM fools BOTH architectures...')
        idxs = find_attack_successes(
            [resnet, mobilenet], pool_images, pool_labels,
            fgsm_attack, FGSM_EPSILON, criterion, DEVICE,
            need=3, require_all=True
        )
        if len(idxs) < 3:
            print(f'  Only {len(idxs)} samples fool both models; relaxing to "either model".')
            idxs = find_attack_successes(
                [resnet, mobilenet], pool_images, pool_labels,
                fgsm_attack, FGSM_EPSILON, criterion, DEVICE,
                need=3, require_all=False
            )

        forensic_records = []
        for n, i in enumerate(idxs[:3]):
            lbl = pool_labels[i].item()
            print(f'  Generating forensic panel {n + 1} (true class: {class_names[lbl]})...')
            save_path = os.path.join(OUTPUT_DIR, f'forensic_comparison_sample_{n}.png')
            forensic_records += compare_models_forensic(
                resnet, mobilenet, pool_images[i], lbl, class_names,
                fgsm_attack, FGSM_EPSILON, resnet_target_layer, mobilenet_target_layer,
                criterion, DEVICE, save_path=save_path
            )

        print(f'  Generated {len(idxs[:3])} forensic panels.')
        for r in forensic_records:
            print(f"    {r['row']:<14} {r['true_class']} -> {r['adv_pred']}  "
                  f"({'FOOLED' if r['attack_succeeded'] else 'RESISTED'})")

        # Keep the chosen samples for the baseline-vs-hardened comparison below
        forensic_pool = (pool_images, pool_labels, idxs)
    except Exception as e:
        print(f"Error in Phase 3/4: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"Phase 3/4 completed in {time.time() - phase34_start:.2f} seconds.")

    # ===========================================================
    # PHASE 5: Model Hardening & The Showdown
    # ===========================================================
    phase5_start = time.time()
    print('\n' + '=' * 50)
    print('  PHASE 5: The Countermeasure Protocol')
    print('=' * 50)

    try:
        # Adversarial training - ResNet-34
        print('\n--- Adversarial Training: ResNet-34 ---')
        resnet_hardened = build_resnet34(num_classes, DEVICE)
        resnet_hardened, resnet_adv_history = adversarial_train_model(
            resnet_hardened, train_loader, val_loader, ADV_EPOCHS, LR,
            fgsm_attack, FGSM_EPSILON, DEVICE, 'ResNet34_Hardened', OUTPUT_DIR
        )
        plot_training_history(resnet_adv_history, 'ResNet34_Hardened', OUTPUT_DIR)

        # Adversarial training - MobileNetV2
        print('\n--- Adversarial Training: MobileNetV2 ---')
        mobilenet_hardened = build_mobilenetv2(num_classes, DEVICE)
        mobilenet_hardened, mobilenet_adv_history = adversarial_train_model(
            mobilenet_hardened, train_loader, val_loader, ADV_EPOCHS, LR,
            fgsm_attack, FGSM_EPSILON, DEVICE, 'MobileNetV2_Hardened', OUTPUT_DIR
        )
        plot_training_history(mobilenet_adv_history, 'MobileNetV2_Hardened', OUTPUT_DIR)

        # THE SHOWDOWN
        print('\n' + '=' * 50)
        print('  THE SHOWDOWN')
        print('=' * 50)

        models_dict = {
            'ResNet-34 Baseline': resnet,
            'ResNet-34 Hardened': resnet_hardened,
            'MobileNetV2 Baseline': mobilenet,
            'MobileNetV2 Hardened': mobilenet_hardened,
        }

        results = run_showdown(models_dict, test_loader, criterion, DEVICE, FGSM_EPSILON, PGD_EPSILON, PGD_ALPHA, PGD_STEPS)
        print_results_table(results)
        plot_results_chart(results, os.path.join(OUTPUT_DIR, 'showdown_results.png'))
        save_results_csv(results, os.path.join(OUTPUT_DIR, 'showdown_results.csv'))

        # -------------------------------------------------------------
        # PHASE 4b: Baseline vs Hardened XAI under the SAME attack
        # Required deliverable: "Saliency/Grad-CAM maps for the original model
        # vs. the hardened model when faced with the same adversarial attack."
        # -------------------------------------------------------------
        print('\n--- Baseline vs Hardened forensic comparison ---')
        pool_images, pool_labels, idxs = forensic_pool
        resnet_target_layer = resnet.layer4[-1]
        resnet_h_target_layer = resnet_hardened.layer4[-1]
        mobilenet_target_layer = mobilenet.features[-1]
        mobilenet_h_target_layer = mobilenet_hardened.features[-1]

        hardening_records = []
        for n, i in enumerate(idxs[:3]):
            lbl = pool_labels[i].item()
            hardening_records += compare_two_models_forensic(
                resnet, resnet_target_layer, 'ResNet-34 Baseline',
                resnet_hardened, resnet_h_target_layer, 'ResNet-34 Hardened',
                pool_images[i], lbl, class_names, fgsm_attack, FGSM_EPSILON,
                criterion, DEVICE,
                title='Baseline vs Hardened under identical FGSM attack (ResNet-34)',
                save_path=os.path.join(OUTPUT_DIR, f'hardening_resnet_sample_{n}.png'))
            hardening_records += compare_two_models_forensic(
                mobilenet, mobilenet_target_layer, 'MobileNetV2 Baseline',
                mobilenet_hardened, mobilenet_h_target_layer, 'MobileNetV2 Hardened',
                pool_images[i], lbl, class_names, fgsm_attack, FGSM_EPSILON,
                criterion, DEVICE,
                title='Baseline vs Hardened under identical FGSM attack (MobileNetV2)',
                save_path=os.path.join(OUTPUT_DIR, f'hardening_mobilenet_sample_{n}.png'))

        for r in hardening_records:
            print(f"    {r['row']:<22} {r['true_class']} -> {r['adv_pred']}  "
                  f"({'FOOLED' if r['attack_succeeded'] else 'RESISTED'})")

        # -------------------------------------------------------------
        # Transferability matrix (was claimed in the report but never measured)
        # -------------------------------------------------------------
        print('\n--- Attack transferability matrix (FGSM) ---')
        transfer = run_transfer_matrix(models_dict, test_loader, fgsm_attack,
                                       FGSM_EPSILON, criterion, DEVICE)
        import pandas as pd
        tdf = pd.DataFrame(transfer)
        print(tdf.to_string(index=False))
        tdf.to_csv(os.path.join(OUTPUT_DIR, 'transfer_matrix.csv'), index=False)
    except Exception as e:
        print(f"Error in Phase 5: {e}")
        import traceback; traceback.print_exc()
        return

    print(f"Phase 5 completed in {time.time() - phase5_start:.2f} seconds.")

    print('\n' + '=' * 70)
    print('  INVESTIGATION COMPLETE')
    print(f'  Total Time: {time.time() - start_time:.2f} seconds')
    print(f'  All outputs saved to: {OUTPUT_DIR}')
    print('=' * 70)

if __name__ == '__main__':
    main()
