import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# -------------------------------
# STEP 1: Feature Extraction
# -------------------------------

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    
    # Extract pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    return np.hstack((mfcc_mean, pitch_mean, duration))


# STEP 2: Load dataset


def load_dataset(audio_folder, label_folder):
    features = []
    labels = []

    for file_name in os.listdir(audio_folder):
        if not file_name.endswith('.wav'):
            continue

        audio_path = os.path.join(audio_folder, file_name)
        label_path = os.path.join(label_folder, file_name.replace('.wav', '.txt'))

        if not os.path.exists(label_path):
            print(f"Label not found for {file_name}, skipping.")
            continue

        try:
            feature_vector = extract_features(audio_path)
            with open(label_path, 'r') as f:
                label = f.read().strip()

            features.append(feature_vector)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return np.array(features), np.array(labels)

# -------------------------------
# STEP 3: Train Model
# -------------------------------

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model


# -------------------------------
# STEP 4: Predict New Audio
# -------------------------------

def predict_emotion(model, audio_file):
    features = extract_features(audio_file).reshape(1, -1)
    return model.predict(features)[0]



# -------------------------------
# MAIN
# -------------------------------

if __name__ == "__main__":
    dataset_path = 'dog_bark_dataset/'  # e.g., dog_bark_dataset/happy/.wav, dog_bark_dataset/fear/.wav
    print("Loading dataset...")
    X, y = load_dataset('audio','labels')
    
    print("Training model...")
    model = train_model(X, y)

    # Test with a new audio file
    test_file = 'sample_bark.wav'
    emotion = predict_emotion(model, test_file)
    print(f"Predicted Emotion by shyam: {emotion}")
