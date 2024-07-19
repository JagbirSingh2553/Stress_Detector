import collections
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
from imblearn.over_sampling import SMOTE
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_wav(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith('.wav'):
                try:
                    file_path = os.path.join(root, file)
                    file_name, file_extension = os.path.splitext(file)
                    new_file_path = os.path.join(root, file_name + '.wav')
                    audio = AudioSegment.from_file(file_path)
                    audio.export(new_file_path, format='wav')
                    os.remove(file_path)
                    logger.info(f"Converted {file_path} to {new_file_path}")
                except Exception as e:
                    logger.error(f"Error converting {file_path} to WAV: {e}")

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
logger.info(df.info())
logger.info(df.head())

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        zero_crossings = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Calculate intensity as the mean amplitude of the signal
        intensity = np.mean(np.abs(y))
        
        # Calculate speech rate as the number of voiced frames per second
        voiced_frames = librosa.effects.split(y, top_db=20)
        speech_rate = len(voiced_frames) / (len(y) / sr)
        
        features = np.hstack([
            zero_crossings, spectral_centroid, spectral_bandwidth, spectral_rolloff,
            mfccs_mean,
            chroma_mean,
            intensity, speech_rate
        ])
        
        # Ensure feature vector length consistency
        expected_length = 31  # Updated based on the actual feature vector length
        if len(features) != expected_length:
            raise ValueError(f"Expected feature vector length {expected_length}, but got {len(features)}")

        return features
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {e}")
        return None

features = []
labels = []

# Define stressed and not stressed categories
stressed_labels = ["angry", "fear", "sad"]

for file_path in df['file_path']:
    feature_vector = extract_features(file_path)
    if feature_vector is not None:
        features.append(feature_vector)
        label = os.path.basename(file_path).split('.')[0]  # Extract label from file name
        if label in stressed_labels:
            labels.append("stressed")
        else:
            labels.append("not stressed")

X = np.array(features)
y = np.array(labels)

def augment_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        augment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
        ])
        augmented_samples = augment(samples=y, sample_rate=sr)
        return augmented_samples, sr
    except Exception as e:
        logger.error(f"Error augmenting audio from {file_path}: {e}")
        return None, None

augmented_features = []
augmented_labels = []

for file_path in df['file_path']:
    for _ in range(2):
        y_aug, sr_aug = augment_audio(file_path)
        if y_aug is not None:
            feature_vector = extract_features(file_path)
            if feature_vector is not None:
                augmented_features.append(feature_vector)
                label = os.path.basename(file_path).split('.')[0]  # Extract label from file name
                if label in stressed_labels:
                    augmented_labels.append("stressed")
                else:
                    augmented_labels.append("not stressed")

X_augmented = np.vstack([X, np.array(augmented_features)])
y_augmented = np.hstack([y, np.array(augmented_labels)])

# Check class distribution
label_counts = collections.Counter(y_augmented)
print(label_counts)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_augmented, y_augmented)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_resampled)

scaler = StandardScaler()
X = scaler.fit_transform(X_resampled)

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
logger.info(f"Cross-Validation Scores: {cv_scores}")
logger.info(f"Mean CV Score: {cv_scores.mean()}")

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train the model
best_model.fit(X_train, y_train)

# Evaluate on validation set
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_report = classification_report(y_val, y_val_pred, labels=np.unique(y_encoded), target_names=label_encoder.inverse_transform(np.unique(y_encoded)), zero_division=0)
logger.info(f"Validation Accuracy: {val_accuracy}")
logger.info("Validation Classification Report:")
logger.info(val_report)

# Evaluate on test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_report = classification_report(y_test, y_test_pred, labels=np.unique(y_encoded), target_names=label_encoder.inverse_transform(np.unique(y_encoded)), zero_division=0)
logger.info(f"Test Accuracy: {test_accuracy}")
logger.info("Test Classification Report:")
logger.info(test_report)

joblib.dump(best_model, './stress_detector_model.pkl')
joblib.dump(scaler, './scaler.pkl')  # Save the scaler

logger.info("Model and scaler trained and saved successfully.")
