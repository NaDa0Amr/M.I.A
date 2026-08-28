# Adversarial ML Security Investigation — Task 5 (Phase II)

A PyTorch pipeline that fine-tunes pretrained CNNs on Caltech-101, breaks them with
gradient-based adversarial attacks, diagnoses *why* they break using two explainability
methods, and hardens them with two defenses — then quantifies the whole thing.

> **Report:** [`REPORT.md`](REPORT.md) is the primary deliverable.

---

## Results at a glance

| Model | Clean | FGSM (ε=0.03) | PGD (ε=0.03) | FGSM + input transform |
|---|---|---|---|---|
| ResNet-34 Baseline | 98.29% | 63.66% | 3.88% | 89.44% |
| ResNet-34 Hardened | 97.98% | **77.48%** | **16.61%** | **93.63%** |
| MobileNetV2 Baseline | 95.81% | 43.79% | 0.00% | 85.25% |
| MobileNetV2 Hardened | 90.84% | **52.95%** | **0.62%** | 80.59% |

Adversarial training buys +13.8 pp of FGSM robustness on ResNet-34 for a 0.31 pp
clean-accuracy cost. PGD remains devastating against every model — the honest
headline of this investigation.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   
pip install -r requirements.txt
```

Caltech-101 downloads automatically to `data/` on first run (~130 MB).

**Pretrained weights:** the models start from torchvision's ImageNet weights
(`ResNet34_Weights.DEFAULT`, `MobileNet_V2_Weights.DEFAULT`), downloaded automatically
by torchvision on first use. Nothing is trained from scratch — the brief asks for
transfer learning, so all convolutional layers are frozen except the last block.

## Running

**Full pipeline (trains from scratch, ~7 min on an RTX 3050):**

```bash
python main.py
```

**Regenerate every figure and table from the saved checkpoints (no training):**

```bash
python regenerate_outputs.py            # everything
python regenerate_outputs.py --quick    # forensic figures only, ~1 min
python regenerate_outputs.py --skip-sweep
```

`regenerate_outputs.py` is the one to use if `outputs/*.pth` already exist — it
guarantees the figures and the report describe the same weights, and it writes
`outputs/results_summary.json` with every number in machine-readable form.

---

## Project layout

| File | Purpose |
|---|---|
| `config.py` | All hyperparameters and paths |
| `data_pipeline.py` | Caltech-101 loading, top-N class subsetting, splits, DataLoaders |
| `models.py` | ResNet-34 / MobileNetV2 transfer-learning setup |
| `train.py` | Clean training and validation loops |
| `attacks.py` | **Phase 2** — FGSM, PGD, epsilon sweep |
| `explainability.py` | **Phase 3** — vanilla-gradient saliency, Grad-CAM, heatmap overlay |
| `forensics.py` | **Phase 4** — multi-panel forensic figures, attack-success filtering |
| `defense.py` | **Phase 5** — adversarial training, input-transform defense |
| `evaluation.py` | The Showdown, transferability matrix, charts, CSVs |
| `main.py` | End-to-end orchestrator (all five phases) |
| `regenerate_outputs.py` | Rebuild all artefacts from saved checkpoints, no training |

## Outputs

| File | What it shows |
|---|---|
| `showdown_results.csv` / `.png` | Baseline vs hardened across all four conditions |
| `epsilon_sweep.csv`, `FGSM_epsilon_sweep.png`, `PGD_epsilon_sweep.png` | Robustness vs attack strength |
| `forensic_comparison_sample_*.png` | Clean vs attacked XAI, ResNet-34 vs MobileNetV2 |
| `hardening_{resnet,mobilenet}_sample_*.png` | Baseline vs hardened XAI under an identical attack |
| `transfer_matrix.csv` | Black-box transferability between models |
| `*_training_history.png` | Loss/accuracy curves |
| `results_summary.json` | Every number above, machine-readable |

---

## Configuration notes

- `NUM_CLASSES_SUBSET = 20` — trained on the 20 most populous Caltech-101 classes
  rather than all 101, for tractable iteration. See REPORT.md §2.1 for the
  implications; this is a deliberate scope reduction, not an oversight.
- `CLAMP_VALID_RANGE = False` — adversarial examples are clamped in normalized
  space. Set `True` for the stricter threat model that projects back onto
  realizable `[0,1]` pixel images (results become slightly weaker but more honest).
- ε is expressed in **normalized** space. ε=0.03 ≈ 1.7/255 in raw pixel terms.

## References

Goodfellow et al. (2015) *Explaining and Harnessing Adversarial Examples* ·
Madry et al. (2018) *Towards Deep Learning Models Resistant to Adversarial Attacks* ·
Selvaraju et al. (2017) *Grad-CAM* · Simonyan et al. (2014) *Deep Inside Convolutional Networks*
