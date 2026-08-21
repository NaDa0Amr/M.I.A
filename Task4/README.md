# Task4: Implementing a Neural Network

Files added:

- `scratchnn/` - minimal autodiff `Value` and MLP implementation
- `train_scratch.py` - trains the scratch MLP on scikit-learn digits
- `train_pytorch.py` - PyTorch training on MNIST
- `requirements.txt` - python dependencies

Quick start:

1. Create a virtualenv and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run scratch training (may be slow):

```powershell
python train_scratch.py
```

3. Run PyTorch MNIST training:

```powershell
python train_pytorch.py
```
