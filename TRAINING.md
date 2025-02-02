# DeepInfant: Infant Cry Classification Model Training Documentation

## Overview
DeepInfant is a deep learning model for classifying infant cries into different categories using audio processing and neural networks. The model is designed to be deployed on iOS devices and uses a pre-trained model for initial weights.

## Project Structure
DeepInfant/
├── Data/
│   └── v2/
│       ├── belly_pain/
│       ├── burping/
│       ├── cold_hot/
│       ├── discomfort/
│       ├── hungry/
│       ├── lonely/
│       ├── scared/
│       ├── tired/
│       └── unknown/
├── processed_dataset/
│   ├── train/
│   ├── test/
│   └── metadata.csv
├── prepare_dataset.py
├── train.py
└── TRAINING.md

## Pre-trained Model
The model uses pre-trained weights from the iOS deployment model as a starting point. This transfer learning approach helps improve performance and reduce training time.

## Data Preparation

### Dataset Structure
The raw dataset should be organized in the following structure under `Data/v2/`:

### Class Labels
- belly_pain (bp): Belly pain cries
- burping (bu): Burping sounds
- cold_hot (ch): Temperature discomfort
- discomfort (dc): General discomfort
- hungry (hu): Hunger cries
- lonely (lo): Loneliness cries
- scared (sc): Fear-related cries
- tired (ti): Tiredness cries
- unknown (un): Unclassified cries

### Preparing the Dataset
Run:
```bash
python prepare_dataset.py
```

This script:
1. Creates train/test splits (80/20)
2. Resamples all audio to 16kHz
3. Converts files to WAV format
4. Generates metadata.csv
5. Organizes processed files in the processed_dataset directory

## Model Architecture

### CNN-LSTM Hybrid
- CNN layers for feature extraction
- Bi-directional LSTM for temporal modeling
- Squeeze-and-excitation blocks for channel attention
- Final classification layers

### Audio Processing
- Sample rate: 16kHz
- Duration: 7 seconds (padded/trimmed)
- Features: Mel spectrogram
  - n_mels: 80
  - n_fft: 1024
  - hop_length: 256
  - frequency range: 20Hz - 8000Hz

### Data Augmentation
During training:
- Random time shift (-100ms to 100ms)
- Random noise injection (30% probability)

## Training Process

### Setup
```bash
python train.py
```

### Parameters
- Batch size: 32
- Learning rate: 0.001
- Epochs: 50
- Optimizer: Adam
- Loss function: CrossEntropyLoss

### Monitoring
The training process outputs:
- Training loss and accuracy per epoch
- Validation loss and accuracy per epoch
- Progress bars
- Best model checkpoint saving

### Model Output
- Trained model saved as 'deepinfant.pth'
- Best model selected based on validation accuracy

## Results
The model outputs predictions for 9 classes with evaluation metrics:
- Training accuracy
- Validation accuracy
- Loss curves

## Deployment
After training, the model can be converted and deployed to iOS devices using Core ML conversion tools.

## Notes
- Ensure sufficient GPU memory for training
- Monitor validation metrics for overfitting
- Backup trained models regularly
- Consider early stopping if validation metrics plateau
