# voice_emotions.py

import numpy as np
import librosa
import pickle
import threading
import queue
import time
import pyaudio
try:
    from keras.models import load_model
except ImportError:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        raise ImportError("Could not import Keras. Please install it with: pip install keras")
from config import VOICE_EMOTIONS, VOICE_TO_COMMON

class VoiceEmotionDetector:

    def __init__(self, model_path, scaler_path, encoder_path, config_path):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 22050
        self.CHUNK = 1024
        self.RECORD_SECONDS = 3

        try:
            self.model = load_model(model_path)
            print("Voice emotion model loaded successfully.")
        except Exception as e:
            print(f"Error loading voice emotion model: {e}")
            self.model = None

        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None

        try:
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)
            print("Encoder loaded successfully.")
        except Exception as e:
            print(f"Error loading encoder: {e}")
            self.encoder = None

        try:
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
            print("Config loaded successfully.")
        except Exception as e:
            print(f"Error loading config: {e}")
            self.config = None

        self.audio = None

        self.audio_queue = queue.Queue()

        self.recorded_audio = []

        self.is_recording = False

        self.thread = None

        self.lock = threading.Lock()

        self.latest_prediction = None

        self.new_prediction_available = False

        self.stream = None

        print("Voice emotion detector initialized successfully.")

    def get_input_device_index(self):
        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(numdevices):
            device = self.audio.get_device_info_by_host_api_device_index(0, i)
            if device.get('maxInputChannels') > 0:
                print(f"Input Device id {i} - {device.get('name')}")
                return i
        print("No input device found.")
        return None

    def start(self):
        print("Starting voice emotion detection...")
        print("Initializing audio stream...")

        self.audio = pyaudio.PyAudio()

        self.input_device_index = self.get_input_device_index()

        self.is_recording = True
        self.stop_recording_event = threading.Event()

        self.recorded_audio = []

        try:
            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=self.input_device_index,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )

            print("Audio stream initialized successfully!")

            self.thread = threading.Thread(target=self.process_audio, daemon=True)
            self.thread.start()

        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.is_recording = False
            return

    def stop(self):
        self.is_recording = False
        if self.stop_recording_event:
            self.stop_recording_event.set()

        if self.stream is not None:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error closing audio stream: {e}")

        if self.audio is not None:
            try:
                self.audio.terminate()
                self.audio = None
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")

        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

        print("Voice emotion detection stopped.")

    def audio_callback(self, in_data, frame_count, time_info, status):
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        with self.lock:
            self.audio_queue.put(audio_data)
            self.recorded_audio.append(audio_data)
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        while self.is_recording and not self.stop_recording_event.is_set():
            audio_chunks = []
            start_time = time.time()

            while self.is_recording and not self.stop_recording_event.is_set():
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    audio_chunks.append(audio_data)
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= self.RECORD_SECONDS:
                        break
                except queue.Empty:
                    continue

            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)
                features = self.extract_features(audio_data)

                if features is not None:
                    if self.scaler is None or self.encoder is None or self.model is None:
                        print("Voice emotion detection components not properly loaded.")
                        continue

                    try:
                        features_scaled = self.scaler.transform(features.reshape(1, -1))

                        features_reshaped = features_scaled.reshape(1, 1, features_scaled.shape[1])

                        prediction = self.model.predict(features_reshaped, verbose=0)
                        emotion_idx = np.argmax(prediction[0])
                        confidence = prediction[0][emotion_idx] * 100

                        emotion = VOICE_EMOTIONS[emotion_idx]
                        common_emotion = VOICE_TO_COMMON.get(emotion, 'Unknown')

                        with self.lock:
                            self.latest_prediction = (common_emotion, confidence)
                            self.new_prediction_available = True

                    except Exception as e:
                        print(f"\nError during voice prediction: {e}")
                else:
                    print("\nNo features extracted from audio data.")
            else:
                print("\nNo audio chunks received.")

    def extract_features(self, audio_data):
        try:
            audio_data = audio_data.astype(np.float32)
            audio_data = audio_data / np.max(np.abs(audio_data))

            expected_length = int(self.RATE * self.RECORD_SECONDS)
            if len(audio_data) < expected_length:
                audio_data = np.pad(audio_data,
                                    (0, expected_length - len(audio_data)),
                                    'constant')
            elif len(audio_data) > expected_length:
                audio_data = audio_data[:expected_length]

            audio_data = librosa.util.normalize(audio_data)

            mel_features = np.mean(librosa.feature.melspectrogram(
                y=audio_data,
                sr=self.RATE,
                n_mels=128
            ).T, axis=0)

            mfcc_features = np.mean(librosa.feature.mfcc(
                y=audio_data,
                sr=self.RATE,
                n_mfcc=13
            ).T, axis=0)

            features = np.hstack((mel_features, mfcc_features))

            return features
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return None

    def get_latest_prediction(self):
        with self.lock:
            prediction = self.latest_prediction
            self.new_prediction_available = False
            return prediction

    def has_new_prediction(self):
        with self.lock:
            return self.new_prediction_available

    def get_audio_data(self):
        with self.lock:
            if self.recorded_audio:
                return np.concatenate(self.recorded_audio)
            else:
                return np.array([])

    def reset_audio_data(self):
        with self.lock:
            while not self.audio_queue.empty():
                self.audio_queue.get()
            self.recorded_audio = []

    def predict_emotion_from_file(self, audio_file_path):
        try:
            audio_data, _ = librosa.load(audio_file_path, sr=self.RATE)

            features = self.extract_features(audio_data)

            if features is not None:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                features_reshaped = features_scaled.reshape(1, 1, features_scaled.shape[1])

                prediction = self.model.predict(features_reshaped, verbose=0)
                emotion_idx = np.argmax(prediction[0])
                confidence = prediction[0][emotion_idx] * 100

                emotion = VOICE_EMOTIONS[emotion_idx]
                common_emotion = VOICE_TO_COMMON.get(emotion, 'Unknown')

                return common_emotion, confidence
            else:
                print("No features extracted from audio data.")
                return 'Unknown', 0.0

        except Exception as e:
            print(f"Error during voice prediction from file: {e}")
            return 'Unknown', 0.0
