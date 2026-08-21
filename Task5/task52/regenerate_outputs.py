"""
Regenerate all figures and tables from the SAVED checkpoints -- no training.

Why this exists
---------------
The numbers in REPORT.md came from checkpoints that already live in outputs/.
Re-running main.py would retrain from scratch (~minutes) and shift every
number. This script loads those exact checkpoints and rebuilds every artefact,
so the report and the figures are guaranteed to describe the same models.

It also produces the artefacts main.py originally lacked:
  * forensic panels filtered to attacks that ACTUALLY succeed
  * a readable signed-perturbation panel
  * baseline-vs-hardened XAI under an identical attack  (required deliverable)
  * an attack-transferability matrix
  * clean accuracy WITH the input-transform defense (the defense's true cost)

Everything measurable is also written to outputs/results_summary.json so the
report can be written from data instead of from memory.

Usage:
    python regenerate_outputs.py                # everything
    python regenerate_outputs.py --skip-sweep   # skip the epsilon sweeps
    python regenerate_outputs.py --quick        # figures only, no showdown
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import pandas as pd

from config import (DATA_ROOT, OUTPUT_DIR, NUM_CLASSES_SUBSET, BATCH_SIZE,
                    NUM_WORKERS, SEED, DEVICE, FGSM_EPSILON, PGD_EPSILON,
                    PGD_ALPHA, PGD_STEPS, EPSILON_SWEEP)
from utils import set_seed, load_model
from data_pipeline import get_caltech101_loaders
from models import build_resnet34, build_mobilenetv2
from attacks import fgsm_attack, pgd_attack, epsilon_sweep
from forensics import compare_models_forensic, compare_two_models_forensic, find_attack_successes
from evaluation import (run_showdown, print_results_table, plot_results_chart,
                        save_results_csv, run_transfer_matrix)
from main import plot_epsilon_sweep

CHECKPOINTS = {
    'ResNet-34 Baseline':   ('resnet',    'best_ResNet34.pth'),
    'ResNet-34 Hardened':   ('resnet',    'ResNet34_Hardened_best.pth'),
    'MobileNetV2 Baseline': ('mobilenet', 'best_MobileNetV2.pth'),
    'MobileNetV2 Hardened': ('mobilenet', 'MobileNetV2_Hardened_best.pth'),
}


def load_all(num_classes):
    models = {}
    for name, (kind, fname) in CHECKPOINTS.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing checkpoint: {path}\n"
                f"Run `python main.py` once to produce it, then re-run this script."
            )
        model = (build_resnet34 if kind == 'resnet' else build_mobilenetv2)(num_classes, DEVICE)
        model = load_model(model, path)
        model.eval()
        models[name] = model
        print(f'  loaded {name:<22} <- {fname}')
    return models


def target_layer_for(name, model):
    return model.layer4[-1] if 'ResNet' in name else model.features[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-sweep', action='store_true')
    ap.add_argument('--quick', action='store_true', help='figures only')
    args = ap.parse_args()

    t0 = time.time()
    set_seed(SEED)
    criterion = nn.CrossEntropyLoss()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = {'device': str(DEVICE), 'fgsm_epsilon': FGSM_EPSILON,
               'pgd': {'epsilon': PGD_EPSILON, 'alpha': PGD_ALPHA, 'steps': PGD_STEPS}}

    print('=' * 70)
    print('  REGENERATING OUTPUTS FROM SAVED CHECKPOINTS (no training)')
    print('=' * 70)

    train_loader, val_loader, test_loader, class_names, num_classes = get_caltech101_loaders(
        root=DATA_ROOT, num_classes_subset=NUM_CLASSES_SUBSET,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=SEED)
    print(f'Classes ({num_classes}): {class_names}')
    print(f'Test set: {len(test_loader.dataset)} images')
    summary['classes'] = class_names
    summary['split_sizes'] = {'train': len(train_loader.dataset),
                              'val': len(val_loader.dataset),
                              'test': len(test_loader.dataset)}

    print('\n--- Loading checkpoints ---')
    models = load_all(num_classes)
    baseline = {k: v for k, v in models.items() if 'Baseline' in k}

    # ---------------- Forensic panels (attack-success filtered) -------------
    print('\n--- Phase 4: forensic panels ---')
    pool_images, pool_labels = [], []
    for bi, (bimg, blbl) in enumerate(test_loader):
        pool_images.append(bimg); pool_labels.append(blbl)
        if bi >= 2:
            break
    pool_images = torch.cat(pool_images)
    pool_labels = torch.cat(pool_labels)
    print(f'  candidate pool: {len(pool_images)} images')

    resnet_b = models['ResNet-34 Baseline']
    mobilenet_b = models['MobileNetV2 Baseline']

    idxs = find_attack_successes([resnet_b, mobilenet_b], pool_images, pool_labels,
                                 fgsm_attack, FGSM_EPSILON, criterion, DEVICE,
                                 need=3, require_all=True)
    mode = 'both architectures fooled'
    if len(idxs) < 3:
        print(f'  only {len(idxs)} fool both; relaxing to "either model fooled"')
        idxs = find_attack_successes([resnet_b, mobilenet_b], pool_images, pool_labels,
                                     fgsm_attack, FGSM_EPSILON, criterion, DEVICE,
                                     need=3, require_all=False)
        mode = 'at least one architecture fooled'
    print(f'  selection criterion: {mode}; chose indices {idxs[:3]}')

    records = []
    for n, i in enumerate(idxs[:3]):
        lbl = pool_labels[i].item()
        out = os.path.join(OUTPUT_DIR, f'forensic_comparison_sample_{n}.png')
        records += compare_models_forensic(
            resnet_b, mobilenet_b, pool_images[i], lbl, class_names,
            fgsm_attack, FGSM_EPSILON,
            target_layer_for('ResNet', resnet_b), target_layer_for('Mobile', mobilenet_b),
            criterion, DEVICE, save_path=out)
        print(f'  wrote {os.path.basename(out)}')
    summary['forensic_panels'] = {'selection': mode, 'records': records}
    for r in records:
        print(f"    {r['row']:<14} {r['true_class']:<12} -> {r['adv_pred']:<12} "
              f"{'FOOLED' if r['attack_succeeded'] else 'RESISTED'}")

    # ---------------- Baseline vs Hardened XAI (required deliverable) -------
    print('\n--- Phase 4b: baseline vs hardened under identical attack ---')
    hardening = []
    pairs = [('ResNet-34 Baseline', 'ResNet-34 Hardened', 'resnet'),
             ('MobileNetV2 Baseline', 'MobileNetV2 Hardened', 'mobilenet')]
    for n, i in enumerate(idxs[:3]):
        lbl = pool_labels[i].item()
        for base_name, hard_name, tag in pairs:
            mb, mh = models[base_name], models[hard_name]
            out = os.path.join(OUTPUT_DIR, f'hardening_{tag}_sample_{n}.png')
            hardening += compare_two_models_forensic(
                mb, target_layer_for(base_name, mb), base_name,
                mh, target_layer_for(hard_name, mh), hard_name,
                pool_images[i], lbl, class_names, fgsm_attack, FGSM_EPSILON,
                criterion, DEVICE,
                title=f'Baseline vs Hardened under identical FGSM attack (eps={FGSM_EPSILON})',
                save_path=out)
            print(f'  wrote {os.path.basename(out)}')
    summary['hardening_panels'] = hardening
    for r in hardening:
        print(f"    {r['row']:<22} {r['true_class']:<12} -> {r['adv_pred']:<12} "
              f"{'FOOLED' if r['attack_succeeded'] else 'RESISTED'}")

    if args.quick:
        _finish(summary, t0)
        return

    # ---------------- Epsilon sweeps ---------------------------------------
    if not args.skip_sweep:
        print('\n--- Phase 2: epsilon sweeps ---')
        sweeps = {}
        for attack, aname, kw in [(fgsm_attack, 'FGSM', {}),
                                  (pgd_attack, 'PGD', dict(alpha=PGD_ALPHA, num_steps=PGD_STEPS))]:
            r = epsilon_sweep(resnet_b, test_loader, attack, EPSILON_SWEEP, criterion, DEVICE, **kw)
            m = epsilon_sweep(mobilenet_b, test_loader, attack, EPSILON_SWEEP, criterion, DEVICE, **kw)
            plot_epsilon_sweep(r, m, aname, OUTPUT_DIR)
            sweeps[aname] = {'ResNet-34': {str(k): round(v * 100, 2) for k, v in r.items()},
                             'MobileNetV2': {str(k): round(v * 100, 2) for k, v in m.items()}}
        summary['epsilon_sweeps'] = sweeps
        pd.DataFrame(
            [{'attack': a, 'model': mdl, 'epsilon': e, 'accuracy_pct': v}
             for a, d in sweeps.items() for mdl, dd in d.items() for e, v in dd.items()]
        ).to_csv(os.path.join(OUTPUT_DIR, 'epsilon_sweep.csv'), index=False)
        print('  wrote epsilon_sweep.csv')

    # ---------------- Showdown ---------------------------------------------
    print('\n--- Phase 5: the showdown ---')
    results = run_showdown(models, test_loader, criterion, DEVICE,
                           FGSM_EPSILON, PGD_EPSILON, PGD_ALPHA, PGD_STEPS)
    print_results_table(results)
    plot_results_chart(results, os.path.join(OUTPUT_DIR, 'showdown_results.png'))
    save_results_csv(results, os.path.join(OUTPUT_DIR, 'showdown_results.csv'))
    summary['showdown'] = results

    # Derived deltas -- so the report never has to eyeball them again
    df = pd.DataFrame(results).set_index('Model')
    deltas = {}
    for arch in ['ResNet-34', 'MobileNetV2']:
        b, h = f'{arch} Baseline', f'{arch} Hardened'
        deltas[arch] = {
            col.replace(' (%)', '') + ' delta_pp': round(df.loc[h, col] - df.loc[b, col], 2)
            for col in df.columns
        }
    summary['hardening_deltas_pp'] = deltas
    print('Hardening deltas (percentage points):')
    print(json.dumps(deltas, indent=2))

    # ---------------- Transferability --------------------------------------
    print('\n--- Attack transferability (FGSM) ---')
    transfer = run_transfer_matrix(baseline, test_loader, fgsm_attack,
                                   FGSM_EPSILON, criterion, DEVICE)
    tdf = pd.DataFrame(transfer)
    print(tdf.to_string(index=False))
    tdf.to_csv(os.path.join(OUTPUT_DIR, 'transfer_matrix.csv'), index=False)
    summary['transfer_matrix'] = transfer

    _finish(summary, t0)


def _finish(summary, t0):
    path = os.path.join(OUTPUT_DIR, 'results_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nWrote {path}')
    print(f'Done in {time.time() - t0:.1f}s. All outputs in {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
