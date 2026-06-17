# 🎤 Audio Keyword Recognition using Deep Learning

A deep learning-based speech command recognition system that classifies short audio clips into predefined keywords using **Log-Mel Spectrograms and a Convolutional Neural Network (CNN)**.

---

## 📌 Project Overview

Human speech is naturally expressed in audio waveforms, but machines cannot directly interpret raw signals. This project transforms audio into **time-frequency representations (spectrograms)** and uses a CNN to classify spoken commands.

We classify:
- 10 spoken commands:  
  `yes, no, up, down, left, right, on, off, stop, go`
- `silence`
- `unknown`

---

## 🚀 Key Features

- 🎧 Converts raw audio into **Log-Mel spectrograms**
- 🧠 Custom CNN for speech command classification
- ⚡ Lightweight architecture (~130K parameters)
- 📊 Strong evaluation using **Macro-F1 + Accuracy**
- 🔁 Comparison with MobileNetV2 transfer learning baseline
- 📉 Handles class imbalance using proper evaluation metrics

---

## 📂 Dataset

- **Google Speech Commands v0.01**
- ~65,000 one-second audio clips
- 10 command words + silence + unknown
- Final split:
  - Train: 45,334 samples  
  - Validation: 9,683 samples  
  - Test: 9,710 samples  

---

## 🔧 Data Preprocessing

1. Load audio files
2. Resample to 1-second clips
3. Pad / trim audio signals
4. Convert to **Log-Mel Spectrogram**
5. Normalize to `[0, 1]`
6. Output shape: **64 × 32 × 1**

---

## 🧠 Model Architecture

Custom CNN:

- Conv2D (32 filters)
- Conv2D (64 filters)
- Conv2D (128 filters)
- Batch Normalization
- Dropout (0.25–0.4)
- Global Average Pooling
- Softmax (12 classes)

Optimizer: **Adam (lr = 0.001)**  
Loss: **Sparse Categorical Crossentropy**

---

## 🏋️ Training Details

- Batch size: 64
- Epochs: 15
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint

---

## 📊 Results

### 🏆 Custom CNN
- Accuracy: **95.1%**
- Macro-F1: **0.85**
- Test Loss: **0.1607**

### ⚠️ MobileNetV2 (Frozen)
- Accuracy: **63.4%**
- Macro-F1: ~0.065  
- Model collapsed to majority class (“unknown”)

---

## 📉 Key Insights

- CNN significantly outperforms transfer learning (ImageNet features are not suitable for spectrograms)
- Model rarely confuses commands with each other
- Most errors go into “unknown” class → safer failure mode
- Class imbalance heavily impacts evaluation → Macro-F1 is essential

---

## ⚠️ Limitations

- Severe class imbalance (unknown dominates dataset)
- Silence class underrepresented
- Limited vocabulary (only 10 commands)
- No real-world noise testing
- MobileNet not fine-tuned (only frozen baseline)

---

## 🔮 Future Work

- Improve silence class using noise slicing
- Apply SpecAugment for robustness
- Test with real-world noisy audio
- Fine-tune pretrained audio models (AudioSet-based)
- Deploy real-time keyword spotting system

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- Scikit-learn
- NumPy
- Matplotlib

---

## 👨‍🎓 Authors

- Abdul Sami  
- Samuele Urosevic  
- Thilina Lakshan  

---

## 📌 Summary

This project demonstrates that a well-designed CNN trained on spectrograms can outperform generic transfer learning models for audio keyword recognition, achieving **95%+ accuracy with strong generalization performance**.
