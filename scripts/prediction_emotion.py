import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0 = all, 1 = info, 2 = warning, 3 = error)

import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys

# Load model and label encoder
model_path = "model/emotionn_modell.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Ensure 'emotionn_modell.h5' exists in the 'model' folder.")
model = load_model(model_path)


emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_pad_len=200):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_pad_len]
    mfcc = np.expand_dims(mfcc, axis=0)  # add batch dim
    return mfcc

# Example usage: python predict_emotion.py path/to/audio.wav
if __name__ == "__main__":
    audio_file = sys.argv[1]
    features = extract_mfcc(audio_file)
    prediction = model.predict(features)
    predicted_label = emotion_labels[np.argmax(prediction)]
    print(f"Predicted Emotion: {predicted_label}")
