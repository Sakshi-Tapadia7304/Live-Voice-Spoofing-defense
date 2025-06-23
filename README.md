# 🎙️ Live Voice Spoofing Defense

With the rise of AI-generated speech, traditional voice authentication systems face growing risks of spoofing. This project presents a real-time spoofing detection system capable of distinguishing between genuine and synthetic voices using acoustic features and machine learning techniques.

## 🧠 Overview

This system analyzes speech samples to detect synthetic/fake voices using:
- **Mel-Frequency Cepstral Coefficients (MFCC)**
- **Constant Q Cepstral Coefficients (CQCC)**
- **Spectrogram Patterns**

Machine learning models are trained using the **ASVspoof 2019 Logical Access** dataset to achieve high accuracy and robustness against spoofing attempts.

## 🚀 Features
- Real-time classification of speech samples as genuine or spoofed
- Preprocessing pipeline with acoustic feature extraction
- Support for multiple model architectures (CNN, Logistic Regression, etc.)
- Evaluation using metrics like Accuracy, Precision, Recall

## 🔧 Tech Stack

**Packages:**
- NumPy
- Matplotlib
- Scikit-learn

**Libraries:**
1. Librosa
2. Torchaudio
3. PyTorch / TensorFlow
4. Soundfile

## 📁 Dataset

- ASVspoof 2019 Logical Access (LA) subset  
- [Download Link](https://www.asvspoof.org/database/)

## 🛠️ Installation

Clone this repo and install the dependencies:
```bash
git clone https://github.com/Sakshi-Tapadia7304/Live-Voice-Spoofing-defense.git
cd Live-Voice-Spoofing-defense
pip install -r requirements.txt
