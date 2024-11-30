# config.py

import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACE_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
VOICE_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disguist', 'surprised']
COMMON_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

VOICE_TO_COMMON = {
    'neutral': 'neutral',
    'calm': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fearful': 'fear',
    'disguist': 'disgust',
    'surprised': 'surprise'
}

DATASET_DIR = "collected_data"
LOGS_DIR = "logs"
MODEL_DIR = "models"

for directory in [DATASET_DIR, LOGS_DIR, MODEL_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

LOG_FILE = os.path.join(LOGS_DIR, "emotion_predictions.csv")

if not os.path.exists(LOG_FILE):
    import csv
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Image_File', 'Audio_File',
                         'Face_Emotion', 'Face_Confidence',
                         'Voice_Emotion', 'Voice_Confidence',
                         'Fused_Emotion', 'User_Feedback'])

FACE_MODEL_PATH = os.path.join(MODEL_DIR, "vit_emotion_model.pth")
PROTOTXT_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
FACE_DETECTOR_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
VOICE_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.pkl")
