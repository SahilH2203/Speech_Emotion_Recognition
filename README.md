# Speech Emotion Recognition from Audio

## 📌 Project Description
This project performs emotion classification using speech audio. It extracts MFCC features from audio files and uses a deep RNN model with LSTM, GRU, and Attention.

## 🔧 Preprocessing
- Audio Resampling to 22,050 Hz
- MFCC Extraction (40 coefficients)
- Padding/Truncating to 200 time steps
- Label Encoding and One-Hot Encoding

## 🧠 Model Pipeline
- Input: MFCC (200, 40)
- BiLSTM + BiGRU layers
- Attention Layer
- Dense + Dropout
- Output: Softmax over 8 emotion classes

## 🎯 Accuracy Metrics
- Test Accuracy: **83.1%**
- Per-Class Accuracy: all > 75%
- F1 Score (macro): **83%**

## 🧪 How to Run
