import torch
import os

# Paths
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  
NUM_CLASSES_SUBSET = 20  
SEED = 42


LR = 1e-4
CLEAN_EPOCHS = 8
ADV_EPOCHS = 5

# Attacks
EPSILON_SWEEP = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
FGSM_EPSILON = 0.03
PGD_EPSILON = 0.03
PGD_ALPHA = 0.007
PGD_STEPS = 7
# Project adversarial examples back onto realizable images ([0,1] pixels).
# False reproduces the numbers in REPORT.md; True is the stricter threat model.
CLAMP_VALID_RANGE = False

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
