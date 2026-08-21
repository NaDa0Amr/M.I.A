
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from explainability import compute_saliency, compute_gradcam, overlay_heatmap
from utils import tensor_to_image

BG = '#1a1a2e'


def get_prediction(model, image, device) -> Tuple[int, float]:
    """Return (predicted_class_index, softmax_confidence) for a single CHW image."""
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        probs = F.softmax(output, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return pred.item(), conf.item()


def noise_to_image(noise: torch.Tensor) -> np.ndarray:

    n = noise.detach().cpu().float()
    if n.dim() == 4:
        n = n[0]
    scale = n.abs().max()
    if scale < 1e-12:
        scale = torch.tensor(1e-12)
    n = (n / scale) * 0.5 + 0.5
    return n.permute(1, 2, 0).numpy().clip(0.0, 1.0)


def noise_stats(noise: torch.Tensor) -> str:
    """Short human-readable summary of perturbation size, for figure captions."""
    n = noise.detach().cpu().float().flatten()
    return f"L_inf={n.abs().max():.4f}  L2={n.norm():.2f}"


def _forensic_row(fig, gs_row, n_rows, model, target_layer, row_label,
                  image, label, class_names, attack_fn, epsilon,
                  criterion, device):
    """
    Draw one 5-column forensic row for a single model. Returns a dict of the
    numbers produced, so callers can report them rather than guess at them.
    """
    clean_pred, clean_conf = get_prediction(model, image, device)
    clean_sal = compute_saliency(model, image, clean_pred, device)
    clean_cam = compute_gradcam(model, image, clean_pred, target_layer, device)
    clean_np = tensor_to_image(image)
    clean_cam_overlay = overlay_heatmap(clean_np, clean_cam)

    if attack_fn is not None:
        adv_image = attack_fn(
            model, image.unsqueeze(0), torch.tensor([label]),
            epsilon, criterion, device
        ).squeeze(0)
    else:
        adv_image = image.clone()

    adv_pred, adv_conf = get_prediction(model, adv_image, device)
    adv_sal = compute_saliency(model, adv_image, adv_pred, device)
    adv_cam = compute_gradcam(model, adv_image, adv_pred, target_layer, device)
    adv_np = tensor_to_image(adv_image)
    adv_cam_overlay = overlay_heatmap(adv_np, adv_cam)

    noise = adv_image - image.to(device)
    noise_np = noise_to_image(noise)

    attack_worked = (adv_pred != label)
    verdict = 'FOOLED' if attack_worked else 'RESISTED'
    verdict_colour = '#ff5370' if attack_worked else '#7bd88f'

    r = gs_row

    ax_clean = fig.add_subplot(n_rows, 5, r * 5 + 1)
    ax_clean.imshow(clean_np)
    ax_clean.set_title(
        f'{row_label}\nClean: {class_names[clean_pred]}\nConf: {clean_conf:.2%}',
        color='white', fontname='monospace', fontsize=10)
    ax_clean.axis('off')

    ax_sal = fig.add_subplot(n_rows * 2, 5, r * 10 + 2)
    ax_sal.imshow(clean_sal, cmap='hot')
    ax_sal.set_title('Clean Saliency', color='white', fontsize=9, fontname='monospace')
    ax_sal.axis('off')

    ax_cam = fig.add_subplot(n_rows * 2, 5, r * 10 + 7)
    ax_cam.imshow(clean_cam_overlay)
    ax_cam.set_title('Clean Grad-CAM', color='white', fontsize=9, fontname='monospace')
    ax_cam.axis('off')

    ax_noise = fig.add_subplot(n_rows, 5, r * 5 + 3)
    ax_noise.imshow(noise_np)
    ax_noise.set_title(
        f'Perturbation (signed)\n{noise_stats(noise)}',
        color='white', fontname='monospace', fontsize=9)
    ax_noise.axis('off')

    ax_adv = fig.add_subplot(n_rows, 5, r * 5 + 4)
    ax_adv.imshow(adv_np)
    ax_adv.set_title(
        f'Adversarial: {class_names[adv_pred]}\nConf: {adv_conf:.2%}\n[{verdict}]',
        color=verdict_colour, fontname='monospace', fontsize=10)
    ax_adv.axis('off')

    ax_asal = fig.add_subplot(n_rows * 2, 5, r * 10 + 5)
    ax_asal.imshow(adv_sal, cmap='hot')
    ax_asal.set_title('Adv Saliency', color='white', fontsize=9, fontname='monospace')
    ax_asal.axis('off')

    ax_acam = fig.add_subplot(n_rows * 2, 5, r * 10 + 10)
    ax_acam.imshow(adv_cam_overlay)
    ax_acam.set_title('Adv Grad-CAM', color='white', fontsize=9, fontname='monospace')
    ax_acam.axis('off')

    return {
        'row': row_label,
        'true_class': class_names[label],
        'clean_pred': class_names[clean_pred],
        'clean_conf': round(clean_conf, 4),
        'adv_pred': class_names[adv_pred],
        'adv_conf': round(adv_conf, 4),
        'attack_succeeded': bool(attack_worked),
        'l_inf': float(noise.abs().max()),
    }


def plot_forensic_panel(model, image, label, class_names, attack_fn, epsilon,
                        target_layer, model_name, criterion, device,
                        save_path=None):
    """Single-model 5-panel forensic dashboard."""
    fig = plt.figure(figsize=(20, 5), facecolor=BG)
    attack_name = attack_fn.__name__ if attack_fn else 'None'
    fig.suptitle(f'Forensic Analysis: {model_name}  (Attack: {attack_name}, eps={epsilon})',
                 color='white', fontsize=16, fontname='monospace')

    record = _forensic_row(fig, 0, 1, model, target_layer, model_name,
                           image, label, class_names, attack_fn, epsilon,
                           criterion, device)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, facecolor=BG, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return [record]


def compare_two_models_forensic(model_a, layer_a, name_a,
                                model_b, layer_b, name_b,
                                image, label, class_names,
                                attack_fn, epsilon, criterion, device,
                                title=None, save_path=None):
    """
    Two-row forensic comparison of any two models under the same attack.

    Generic on purpose: used both for architecture-vs-architecture
    (ResNet-34 vs MobileNetV2) and for the baseline-vs-hardened comparison
    the brief asks for explicitly.
    """
    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    attack_name = attack_fn.__name__ if attack_fn else 'None'
    fig.suptitle(title or f'Forensic Comparison: {name_a} vs {name_b}  '
                          f'({attack_name}, eps={epsilon})',
                 color='white', fontsize=18, fontname='monospace')

    records = []
    for row, (model, layer, name) in enumerate(
            [(model_a, layer_a, name_a), (model_b, layer_b, name_b)]):
        records.append(
            _forensic_row(fig, row, 2, model, layer, name, image, label,
                          class_names, attack_fn, epsilon, criterion, device)
        )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, facecolor=BG, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return records


def compare_models_forensic(resnet_model, mobilenet_model, image, label, class_names,
                            attack_fn, epsilon, resnet_target_layer,
                            mobilenet_target_layer, criterion, device, save_path=None):
    """Backwards-compatible wrapper around compare_two_models_forensic."""
    return compare_two_models_forensic(
        resnet_model, resnet_target_layer, 'ResNet-34',
        mobilenet_model, mobilenet_target_layer, 'MobileNetV2',
        image, label, class_names, attack_fn, epsilon, criterion, device,
        title='Architecture Comparison: ResNet-34 vs MobileNetV2',
        save_path=save_path,
    )


def find_attack_successes(models, image_batch, label_batch, attack_fn, epsilon,
                          criterion, device, need=3, require_all=True):
    """
    Select samples that are (a) correctly classified when clean and
    (b) actually misclassified after the attack.

    Without (b), a forensic panel can end up documenting an attack that never
    worked -- which is exactly what the brief's Phase 4 asks you not to do.

    Args:
        models: list of models the condition must hold for
        require_all: if True, every model must be fooled; if False, any one
    Returns:
        list of indices into the batch
    """
    chosen = []
    for i in range(len(image_batch)):
        img = image_batch[i]
        lbl = label_batch[i].item()

        clean_ok, fooled = [], []
        for m in models:
            p, _ = get_prediction(m, img, device)
            clean_ok.append(p == lbl)
            adv = attack_fn(m, img.unsqueeze(0), torch.tensor([lbl]),
                            epsilon, criterion, device).squeeze(0)
            ap, _ = get_prediction(m, adv, device)
            fooled.append(ap != lbl)

        if not all(clean_ok):
            continue
        if (all(fooled) if require_all else any(fooled)):
            chosen.append(i)
        if len(chosen) >= need:
            break
    return chosen
