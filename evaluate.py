# evaluate.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import LOG_FILE, FACE_EMOTIONS, VOICE_EMOTIONS

def visualize_emotion_distributions():
    if not os.path.exists(LOG_FILE):
        print("No log file found.")
        return

    df = pd.read_csv(LOG_FILE)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.countplot(data=df, x='Face_Emotion', order=FACE_EMOTIONS)
    plt.title('Face Emotion Distribution')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    sns.countplot(data=df, x='Voice_Emotion', order=VOICE_EMOTIONS)
    plt.title('Voice Emotion Distribution')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    fused_emotions_order = FACE_EMOTIONS
    sns.countplot(data=df, x='Fused_Emotion', order=fused_emotions_order)
    plt.title('Fused Emotion Distribution')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def visualize_confidence_scores():
    if not os.path.exists(LOG_FILE):
        print("No log file found.")
        return

    df = pd.read_csv(LOG_FILE)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df['Face_Confidence'].astype(float), bins=20, kde=True, color='blue')
    plt.title('Face Emotion Confidence Distribution')
    plt.xlabel('Confidence (%)')

    plt.subplot(1, 2, 2)
    sns.histplot(df['Voice_Confidence'].astype(float), bins=20, kde=True, color='green')
    plt.title('Voice Emotion Confidence Distribution')
    plt.xlabel('Confidence (%)')

    plt.tight_layout()
    plt.show()

def visualize_user_feedback():
    if not os.path.exists(LOG_FILE):
        print("No log file found.")
        return

    df = pd.read_csv(LOG_FILE)

    if 'User_Feedback' not in df.columns or df['User_Feedback'].isnull().all():
        print("No user feedback available.")
        return

    feedback = df.dropna(subset=['User_Feedback'])

    emotions_order = FACE_EMOTIONS

    plt.figure(figsize=(10, 5))
    sns.countplot(data=feedback, x='User_Feedback', order=emotions_order)
    plt.title('User Feedback Emotion Distribution')
    plt.xticks(rotation=45)
    plt.show()

def evaluate_real_time_dataset():
    while True:
        print("\nEvaluation Options:")
        print("1. Visualize Emotion Distributions")
        print("2. Visualize Confidence Score Distributions")
        print("3. Visualize User Feedback")
        print("4. Back to Main Menu")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            visualize_emotion_distributions()
        elif choice == '2':
            visualize_confidence_scores()
        elif choice == '3':
            visualize_user_feedback()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")
