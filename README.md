# ACDC (Audio Consistency Deepfake Checker)

Detect AI-generated speech by analyzing physical inconsistencies in acoustic signals. This project uses physics-based acoustic features rather than model-specific artifacts to distinguish authentic speech from synthetic audio across multiple voice cloning systems.

Tanay Agrawal, Vihaan Byuhatti, Akshith Chalamalasetty, Ashton Tovar Burke

## Core Hypothesis

Real human speech obeys physical constraints imposed by the vocal folds, vocal tract anatomy, and recording environment. Synthetic speech often fails to reproduce these constraints consistently, creating detectable acoustic fingerprints.

## Datasets

**Real speech:**
- LibriSpeech
- ASVspoof 2021
- CSTR VCTK Corpus

**Synthetic speech generated using:**
- xTTS-v2
- OmniVoice
- Custom voice cloning from collected recordings

## Pipeline

### 1. Feature Extraction

Three complementary physics-inspired acoustic representations:

**A. Glottal Vocal Pulse**
- Linear Predictive Coding (LPC) inverse filtering
- Glottal waveform extraction
- Mel spectrogram analysis
- *Hypothesis:* Real vocal fold excitation follows natural physiological patterns; synthetic excitation exhibits unnatural temporal behavior

**B. Vocal Tract Geometry**
- LPC filter coefficients and reflection coefficients
- Tube resonator model with cross-sectional area estimation
- Anatomical realism analysis
- *Hypothesis:* Real speech continuously changes vocal tract shape; synthetic speech produces static or overly smooth estimates

**C. Recording Environment Consistency**
- Blind deconvolution and cepstral analysis
- Room impulse response estimation
- Extracted metrics: impulse response variation, LPC residual kurtosis, spectral centroid, channel drift, noise floor drift
- *Hypothesis:* Authentic recordings exhibit consistent room impulse responses; synthetic audio introduces inconsistent environmental characteristics

### 2. Feature Fusion

Combine all three feature groups into a single numerical feature vector with optional preprocessing (normalization, scaling, PCA, feature selection).

### 3. Machine Learning

Binary classifier trained on fused features.

**Candidate models:**
- CNN
- MLP (feedforward neural network)
- Random Forest
- XGBoost
- SVM (baseline)

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC
- Confusion Matrix

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run feature extraction
python feature_extraction/glottal.py
python feature_extraction/vocal_tract.py
python feature_extraction/room_features.py

# Train and evaluate classifier
python models/train.py
python models/evaluate.py
```

## Technologies

**Core Libraries:**
- PyTorch
- NumPy, SciPy
- Librosa (audio processing)
- scikit-learn (machine learning)
- Matplotlib, Pandas (visualization)
- SoundFile (audio I/O)

**Voice Cloning:**
- OmniVoice
- xTTS-v2

**Signal Processing:**
- LPC (Linear Predictive Coding)
- Cepstral analysis
- Mel spectrograms
- Reflection coefficients
- Blind deconvolution

## Repository Structure

```
cosmos-project-26/
├── datasets/                  # Audio data (real/synthetic pairs)
├── data_processing/
│   ├── preprocessing.py
│   └── generate_dataset.py
├── feature_extraction/
│   ├── glottal.py
│   ├── vocal_tract.py
│   └── room_features.py
├── models/
│   ├── cnn.py
│   ├── mlp.py
│   ├── train.py
│   └── evaluate.py
├── experiments/               # Experiment logs and configurations
├── results/                   # Model outputs and evaluations
├── utils/                     # Utility functions
├── notebooks/                 # Jupyter notebooks (optional)
├── requirements.txt
├── .gitignore
└── README.md
```

## Current Progress

✅ Completed:
- Dataset collection
- Voice cloning pipeline
- LPC glottal extraction
- Vocal tract modeling
- Reflection coefficient analysis
- Preliminary recording environment experiments
- Initial dataset construction

🔄 In Progress:
- OmniVoice synthetic dataset generation
- Full feature extraction pipeline
- Feature fusion
- Machine learning training

📋 Planned:
- End-to-end classifier
- Hyperparameter tuning
- Cross-model generalization experiments
- Comprehensive evaluation on unseen deepfake models

## Research Contribution

Unlike existing deepfake detectors relying on learned embeddings or spectrogram classification alone, this project prioritizes **interpretable, physics-based acoustic representations**. By combining glottal source analysis, vocal tract estimation, and recording environment consistency, the system aims to improve robustness against future voice cloning models while providing insight into classification decisions.
