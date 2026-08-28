---
title: Neural Image Captioning
emoji: 🖼️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.19.0
app_file: app/gradio_app.py
pinned: false
---
# 🖼️ Neural Image Caption Generation

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

## Overview
End-to-end image captioning system using deep learning, combining computer vision (ResNet-50) with NLP (LSTM) to generate natural language descriptions of images.

## Architecture

```mermaid
graph LR
    A[Input Image] -->|Resize & Normalize| B[ResNet-50 Frozen]
    B -->|2048-d| C[Linear Projection]
    C -->|256-d| D[LSTM Decoder]
    D -->|Step-by-step| E[Caption]
```

This model follows the "Show and Tell" architecture by utilizing a pre-trained CNN to extract image features and passing those features as the initial state or input to an LSTM which decodes them into a natural language sequence.

## Dataset
We use the **Flickr8k** dataset:
- **8,092** images in total
- **5** captions per image
- **Splits**: 
  - Train: 6,000 images
  - Val: 1,000 images
  - Test: 1,000 images
- **Preprocessing**: Lowercase conversion, punctuation removal, and frequency thresholding.

## Project Structure
```text
.
├── app/
│   ├── app.py           (Streamlit App)
│   └── gradio_app.py    (Gradio App)
├── checkpoints/
├── scripts/
│   ├── extract_features.py
│   ├── train.py
│   └── evaluate.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── inference/
│   ├── models/
│   └── utils/
├── tests/
│   ├── integration/
│   └── unit/
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd Task62
pip install -r requirements.txt
make setup
```

## Quick Start

1. Download the Flickr8k dataset from Kaggle.
2. Run data preparation: `python setup_data.py`
3. Extract features: `python scripts/extract_features.py`
4. Train the model: `python scripts/train.py`
5. Evaluate: `python scripts/evaluate.py`

## Training
| Hyperparameter | Value |
|----------------|-------|
| Embed Dim      | 256   |
| Hidden Dim     | 512   |
| Batch Size     | 64    |
| Learning Rate  | 1e-3  |

Features include early stopping, LR scheduling, gradient clipping, and automated checkpointing.

## Evaluation
| Metric | Expected Score |
|--------|----------------|
| BLEU-1 | ~60.0          |
| BLEU-4 | ~20.0          |
| ROUGE-L| ~45.0          |
| METEOR | ~22.0          |

*Qualitative examples showing generated captions will be displayed here.*

## Web Interfaces

You have two options for the web interface: **Streamlit** or **Gradio**.

### Option A: Streamlit App
Start the Streamlit interface to upload and test your own images:
```bash
streamlit run app/app.py
# or
make app
```
Available at: `http://localhost:8501`

### Option B: Gradio App
Start the Gradio interface:
```bash
python app/gradio_app.py
# or
make gradio
```
Available at: `http://localhost:7860`

## Docker

Run both the Streamlit UI and Gradio UI inside Docker containers:
```bash
# Build the image
docker build -t image-captioning .

# Run both services
docker-compose up

# Services:
#   Streamlit UI  → http://localhost:8501
#   Gradio UI     → http://localhost:7860
```

## Model on HuggingFace
🤗 Model available at: [HuggingFace Hub Link](https://huggingface.co/username/flickr8k-image-captioning)

## Testing
Run the test suite using pytest:
```bash
pytest tests/ -v --cov=src
```

## Technologies
- Python 3.10+
- PyTorch & torchvision (ResNet-50)
- Streamlit & Gradio (Web UIs)
- Docker & Docker Compose
- NLTK, rouge-score
- HuggingFace Hub

## License
MIT
