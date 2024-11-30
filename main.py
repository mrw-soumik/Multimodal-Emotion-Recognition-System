# main.py

import os
import sys
from face_emotion import FaceEmotionDetector
from voice_emotion import VoiceEmotionDetector
from utils import collect_sample, visualize_real_time_predictions
from evaluate import evaluate_real_time_dataset
from config import FACE_MODEL_PATH, VOICE_MODEL_PATH, SCALER_PATH, ENCODER_PATH, CONFIG_PATH

def main():
    try:
        face_detector = FaceEmotionDetector(model_path=FACE_MODEL_PATH)
    except Exception as e:
        print(f"An error occurred while initializing the face detector: {e}")
        face_detector = None

    try:
        if not os.path.exists(VOICE_MODEL_PATH):
            raise FileNotFoundError(f"Voice model file not found at {VOICE_MODEL_PATH}")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Encoder file not found at {ENCODER_PATH}")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

        voice_detector = VoiceEmotionDetector(
            model_path=VOICE_MODEL_PATH,
            scaler_path=SCALER_PATH,
            encoder_path=ENCODER_PATH,
            config_path=CONFIG_PATH
        )
        if voice_detector.model is None or voice_detector.scaler is None or voice_detector.encoder is None:
            raise ValueError("Voice emotion detector components failed to load.")
        print("Voice emotion detector initialized successfully.")
    except Exception as e:
        print(f"An error occurred while initializing the voice detector: {e}")
        voice_detector = None

    if face_detector is None or voice_detector is None:
        print("Cannot proceed without both face and voice models.")
        sys.exit()

    while True:
        print("\nSelect an option:")
        print("1. Real-Time Emotion Detection")
        print("2. Collect a Sample")
        print("3. Visualize Logged Predictions")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            visualize_real_time_predictions(face_detector, voice_detector)
        elif choice == '2':
            collect_sample(face_detector, voice_detector)
        elif choice == '3':
            evaluate_real_time_dataset()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")

if __name__ == '__main__':
    main()
