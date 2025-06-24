import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

# Load model once
model_path = "model/emotionn_modell.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Ensure 'emotionn_modell.h5' exists in the 'model' folder.")
model = load_model(model_path)

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Feature extractor
def extract_mfcc(file_path, sr=22050, n_mfcc=40, max_pad_len=200):
    y, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    if mfcc.shape[0] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:max_pad_len]
    return np.expand_dims(mfcc, axis=0)

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")

st.markdown("Upload a `.wav` file to classify the emotion expressed in the speech.")

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Save uploaded file to a temp path
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_mfcc("temp.wav")
        prediction = model.predict(features)[0]
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        st.success(f"**Predicted Emotion:** {predicted_emotion}")

        # Show confidence
        st.subheader("Class Probabilities:")
        for emotion, prob in zip(emotion_labels, prediction):
            st.write(f"{emotion}: {prob:.2%}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
