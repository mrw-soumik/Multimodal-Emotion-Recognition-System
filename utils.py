# utils.py

import os
import cv2
import datetime
import csv
from config import DATASET_DIR, LOG_FILE, FACE_EMOTIONS
from multimodel import fuse_emotions
import time
import soundfile as sf

def log_prediction(timestamp, image_path, audio_path,
                   face_emotion, face_confidence,
                   voice_emotion, voice_confidence,
                   fused_emotion):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            image_path,
            audio_path,
            face_emotion if face_emotion else 'Unknown',
            f"{face_confidence:.2f}" if face_confidence else '0.00',
            voice_emotion if voice_emotion else 'Unknown',
            f"{voice_confidence:.2f}" if voice_confidence else '0.00',
            fused_emotion if fused_emotion else 'Unknown',
            ''  # Placeholder for user feedback
        ])
    print("Logged predictions.")

def update_log_with_feedback(timestamp, user_feedback):
    with open(LOG_FILE, 'r', newline='') as f:
        reader = list(csv.reader(f))

    for row in reader:
        if row[0] == timestamp:
            row[-1] = user_feedback
            break

    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(reader)

    print("Updated log with user feedback.")

def get_user_feedback(fused_emotion):
    print("\nWould you like to provide feedback on the predicted emotion?")
    print("1. Yes")
    print("2. No")
    choice = input("Enter your choice (1-2): ")

    if choice == '1':
        print("\nPlease select the true emotion from the following options:")
        emotions_list = FACE_EMOTIONS
        for idx, emotion in enumerate(emotions_list, 1):
            print(f"{idx}. {emotion}")
        print(f"{len(emotions_list)+1}. Unknown/Other")

        while True:
            label_input = input(f"Enter the number corresponding to the true emotion (1-{len(emotions_list)+1}): ")
            try:
                label_idx = int(label_input)
                if 1 <= label_idx <= len(emotions_list):
                    true_emotion = emotions_list[label_idx-1]
                    break
                elif label_idx == len(emotions_list)+1:
                    true_emotion = 'Unknown'
                    break
                else:
                    print(f"Please enter a number between 1 and {len(emotions_list)+1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        return true_emotion
    elif choice == '2':
        return None
    else:
        print("Invalid choice. Skipping feedback.")
        return None

def collect_sample(face_detector, voice_detector):
    while True:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("\nPress 'c' to start sample collection.")
        print("Press 'q' to exit.")

        recording = False
        image_captured = False

        voice_detector.reset_audio_data()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            cv2.imshow('Sample Collection - Press "c" to start, "q" to exit', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and not recording:
                recording = True
                voice_detector.start()
                start_time = time.time()
                print("Sample collection started...")

            if recording:
                if not image_captured:
                    image = frame.copy()
                    image_captured = True
                    print("Image captured.")

                elapsed_time = time.time() - start_time
                if elapsed_time >= voice_detector.RECORD_SECONDS:
                    voice_detector.stop()
                    print("Sample collection completed.")
                    break

            if key == ord('q'):
                if recording:
                    voice_detector.stop()
                print("Exiting sample collection.")
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

        if image_captured:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

            image_path = os.path.join(DATASET_DIR, f"image_{timestamp}.jpg")
            cv2.imwrite(image_path, image)
            print(f"Saved image: {image_path}")
        else:
            print("No image captured.")
            return

        audio_data = voice_detector.get_audio_data()
        if audio_data.size > 0:
            audio_path = os.path.join(DATASET_DIR, f"audio_{timestamp}.wav")
            sf.write(audio_path, audio_data, voice_detector.RATE)
            print(f"Saved audio: {audio_path}")
        else:
            print("No audio data collected.")
            audio_path = None

        face_emotion, face_confidence = face_detector.process_image(image_path)
        if audio_path:
            voice_emotion, voice_confidence = voice_detector.predict_emotion_from_file(audio_path)
        else:
            voice_emotion, voice_confidence = 'Unknown', 0.0

        fused_emotion = fuse_emotions(face_emotion, face_confidence, voice_emotion, voice_confidence)

        log_prediction(timestamp, image_path, audio_path,
                       face_emotion, face_confidence,
                       voice_emotion, voice_confidence,
                       fused_emotion)

        user_feedback = get_user_feedback(fused_emotion)
        if user_feedback:
            update_log_with_feedback(timestamp, user_feedback)

        print("\nWould you like to collect another sample?")
        print("1. Yes")
        print("2. No")
        continue_choice = input("Enter your choice (1-2): ")

        if continue_choice == '1':
            continue
        elif continue_choice == '2':
            print("Exiting sample collection.")
            break
        else:
            print("Invalid choice. Exiting sample collection.")
            break

def visualize_real_time_predictions(face_detector, voice_detector):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    voice_detector.start()

    print("Press 'q' to quit.")

    last_face_emotion = None
    last_face_confidence = None
    last_voice_emotion = None
    last_voice_confidence = None
    last_fused_emotion = None

    last_face_boxes = []
    face_not_found_counter = 0
    face_not_found_threshold = 30

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        face_results = face_detector.process_frame(frame)
        if face_results:
            last_face_boxes = []
            for emotion, confidence, box in face_results:
                x, y, x2, y2 = box
                last_face_boxes.append((emotion, confidence, (x, y, x2, y2)))
                last_face_emotion = emotion
                last_face_confidence = confidence
            face_not_found_counter = 0
        else:
            face_not_found_counter += 1
            if face_not_found_counter >= face_not_found_threshold:
                last_face_boxes = []
                last_face_emotion = None
                last_face_confidence = None

        elapsed_time = time.time() - start_time
        if elapsed_time >= voice_detector.RECORD_SECONDS:
            start_time = time.time()
            if voice_detector.has_new_prediction():
                voice_prediction = voice_detector.get_latest_prediction()
                if voice_prediction:
                    last_voice_emotion, last_voice_confidence = voice_prediction
                    print(f"Voice Emotion Prediction: {last_voice_emotion} ({last_voice_confidence:.1f}%)")

            if last_face_emotion and last_voice_emotion:
                fused_emotion = fuse_emotions(last_face_emotion, last_face_confidence,
                                              last_voice_emotion, last_voice_confidence)
                last_fused_emotion = fused_emotion
                if last_fused_emotion:
                    print(f"Fused Emotion: {last_fused_emotion}")

        for emotion, confidence, box in last_face_boxes:
            x, y, x2, y2 = box
            color = (0, 255, 0) if emotion == "happy" else (0, 0, 255)  # Green for happy, red otherwise
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if last_voice_emotion and last_voice_confidence:
            cv2.putText(frame, f"Voice: {last_voice_emotion} ({last_voice_confidence:.1f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        if last_fused_emotion:
            cv2.putText(frame, f"Fused Emotion: {last_fused_emotion}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    voice_detector.stop()
