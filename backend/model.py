import os
import numpy as np
import pandas as pd
import librosa
from pydub import AudioSegment
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

def convert_to_wav(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.wav'):
                file_path = os.path.join(root, file)
                file_name, file_extension = os.path.splitext(file)
                new_file_path = os.path.join(root, file_name + '.wav')
                audio = AudioSegment.from_file(file_path)
                audio.export(new_file_path, format='wav')
                os.remove(file_path)
                print(f"Converted {file_path} to {new_file_path}")

convert_to_wav('./files')

def list_wav_files(directory):
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    return wav_files

wav_files = list_wav_files('./files')
df = pd.DataFrame(wav_files, columns=['file_path'])
print(df.info())
print(df.head())

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    mfccs_var = np.var(mfccs.T, axis=0)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    chroma_var = np.var(chroma.T, axis=0)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    mel_var = np.var(mel.T, axis=0)
    
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50.0, n_bands=4)
    contrast_mean = np.mean(contrast.T, axis=0)
    contrast_var = np.var(contrast.T, axis=0)
    
    return np.hstack([zero_crossings, spectral_centroid, spectral_bandwidth, spectral_rolloff, mfccs_mean, mfccs_var, chroma_mean, chroma_var, mel_mean, mel_var, contrast_mean, contrast_var])

features = []
labels = []

for file_path in df['file_path']:
    feature_vector = extract_features(file_path)
    features.append(feature_vector)
    label = os.path.basename(os.path.dirname(file_path))
    labels.append(label)

X = np.array(features)
y = np.array(labels)

def augment_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
    ])
    augmented_samples = augment(samples=y, sample_rate=sr)
    return augmented_samples, sr

augmented_features = []
augmented_labels = []

for file_path in df['file_path']:
    for _ in range(2):
        y_aug, sr_aug = augment_audio(file_path)
        feature_vector = extract_features(file_path)
        augmented_features.append(feature_vector)
        label = os.path.basename(os.path.dirname(file_path))
        augmented_labels.append(label)

X_augmented = np.vstack([X, np.array(augmented_features)])
y_augmented = np.hstack([y, np.array(augmented_labels)])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_augmented)

scaler = StandardScaler()
X = scaler.fit_transform(X_augmented)

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10, 12],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X, y_encoded)

best_model = grid_search.best_estimator_

cv_scores = cross_val_score(best_model, X, y_encoded, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean()}")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
labels = np.unique(y_encoded)
target_names = label_encoder.inverse_transform(labels)
report = classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

joblib.dump(best_model, './stress_detector_model.pkl')
joblib.dump(scaler, './scaler.pkl')  # Save the scaler

print("Model and scaler trained and saved successfully.")
