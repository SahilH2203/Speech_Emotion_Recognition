## Project Description
This project focuses on automatic emotion classification from speech audio files. It uses deep learning techniques to analyze the emotional tone in human voice recordings.

The system takes .wav audio inputs, extracts key audio features, and feeds them to a Recurrent Neural Network (RNN) architecture with LSTM, GRU, and Attention Mechanism for final emotion prediction.

The model is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

##  Pre-processing Methodology
#### Audio Loading and Resampling
All audio files are resampled to a uniform sampling rate of 22,050 Hz for consistency.

#### Feature Extraction
Extracted 40-dimensional MFCC (Mel-Frequency Cepstral Coefficients) features from each audio file.
This captures the spectral characteristics of speech relevant for emotion recognition.

#### Sequence Length Handling
Each MFCC feature sequence is padded or truncated to a fixed length of 200 time steps to handle variable-length recordings.

#### Label Encoding and One-Hot Transformation
Emotion labels are encoded using LabelEncoder and then converted into one-hot vectors for classification.

## Model Pipeline

| Stage                   | Details                                                               |
| ----------------------- | --------------------------------------------------------------------- |
| **Input**               | Shape: `(200, 40)` MFCC sequence                                      |
| **Recurrent Layers**    | Bidirectional **LSTM (128 units)** → **GRU (128 units)** with dropout |
| **Attention Mechanism** | To focus on important time steps in the sequence                      |
| **Dense Layers**        | Fully Connected Layer with **ReLU** and **Dropout**                   |
| **Output Layer**        | **Softmax** activation over **8 emotion classes**                     |
| **Loss Function**       | Categorical Crossentropy                                              |
| **Optimizer**           | Adam Optimizer                                                        |

## Accuracy Metric

| Metric                 | Value                         |
| ---------------------- | ----------------------------- |
| **Test Accuracy**      | **83.1%**                     |
| **Per-Class Accuracy** | All emotion classes ≥ **75%** |
| **Macro F1-Score**     | **83%**                       |
| **Dataset Used**       | RAVDESS                       |
#### Per Class Accuracy Breakdown

| Emotion   | Accuracy |
| --------- | -------- |
| Angry     | 80.0%    |
| Calm      | 92.0%    |
| Disgust   | 87.2%    |
| Fearful   | 80.0%    |
| Happy     | 80.0%    |
| Neutral   | 84.2%    |
| Sad       | 78.7%    |
| Surprised | 87.2%    |



