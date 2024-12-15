# Ship Detection in Satellite Imagery

This project focuses on detecting ships in satellite imagery using a convolutional neural network (CNN) model. The dataset is based on the "Ships in Satellite Imagery" dataset, which includes satellite images labeled as "with ship" or "no ship." The goal is to classify images into these two categories.

## Features
- Preprocessing the dataset stored in `shipsnet.json`.
- Building and training a custom CNN model for ship detection.
- Testing and evaluating the model's performance on unseen data.
- Exporting the trained model for future use.

---

## Dataset
- **Source**: [***Ships in Satellite Imagery***](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)
- **Structure**:
  - Images are stored as flattened arrays of size `80x80x3` (RGB).
  - Labels are binary (`1` for "with ship", `0` for "no ship").
- File: `shipsnet.json`

---

## Requirements
The project is built using Python and PyTorch (CUDA 12.4)
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
```bash
pip install -r requirements.txt
```
## Usage
### 1. Preprocess the Dataset
Load and preprocess the data
```python
from preprocess import load_data
X_train, X_test, y_train, y_test = load_data('shipsnet.json')
```
### 2. Train the Model
Run the Notebook to train the CNN model `ship-detection-training.ipynb`.
### 3. Test the Model
Use the trained model to predict images `ship-detection-inference.ipynb`.
