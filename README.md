
# Multimodal Emotion Detection

This project combines **face emotion detection** and **voice emotion detection** into a **multimodal emotion detection system**. Using separate models for each modality, emotions are fused based on confidence scores to provide an accurate and robust emotion recognition system. The system can log predictions, visualize emotion distributions, and accept user feedback for evaluation.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Training](#training)
  - [Face Emotion Detection](#face-emotion-detection)
  - [Voice Emotion Detection](#voice-emotion-detection)
- [Using the Multimodal System](#using-the-multimodal-system)
- [Visualization and Logging](#visualization-and-logging)
- [Future Work](#future-work)

---

## Features

- **Face Emotion Detection**: Vision Transformer (ViT)-based model trained on face images.
- **Voice Emotion Detection**: LSTM-based model trained on MFCC and mel spectrogram features from audio.
- **Multimodal Fusion**: Combines face and voice predictions to determine the final emotion.
- **Logging and Visualization**:
  - Logs predictions for analysis.
  - Visualizes emotion distributions, confidence scores, and user feedback.

---

## Technologies Used

- **Python**
- **Deep Learning Frameworks**: PyTorch, TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Audio Processing**: Librosa, PyAudio
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: NumPy, Pandas

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/multimodal-emotion-detection.git
   cd multimodal-emotion-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the models and supporting files are in the `models/` directory:
   - `vit_emotion_model.pth`: Face emotion detection model.
   - `emotion_model.keras`: Voice emotion detection model.
   - Supporting files: `scaler.pkl`, `encoder.pkl`, `model_config.pkl`, `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`.

---

## Training

### Face Emotion Detection

The face emotion detection model is based on the Vision Transformer (ViT) architecture.

1. **Dataset**: Train the model on the FER2013 dataset, which includes 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

2. **Training Code**:
   - Located in the project directory.
   - Key features:
     - Uses `torchvision.transforms` for data augmentation.
     - Model architecture defined using the `timm` library.
     - Optimized using AdamW and CosineAnnealingLR for learning rate scheduling.

3. **Output**:
   - Trained model saved as `vit_emotion_model.pth`.

### Voice Emotion Detection

The voice emotion detection model uses a stacked LSTM architecture with MFCC and mel spectrogram features.

1. **Datasets**: 
   - **RAVDESS** and **TESS** datasets for emotional speech.
   - Mapped labels to align with common emotion categories.

2. **Feature Extraction**:
   - Extract **MFCC** and **mel spectrogram** features using Librosa.
   - Normalize features with MinMaxScaler.

3. **Training Code**:
   - Located in the project directory.
   - Key features:
     - TimeDistributed Conv1D layers for initial feature extraction.
     - Bidirectional LSTMs for temporal modeling.
     - Dense layers for emotion classification.

4. **Output**:
   - Trained model saved as `emotion_model.keras`.

---

## Using the Multimodal System

1. **Setup**:
   - Ensure all required models and files are in the `models/` directory.
   - Supported files:
     - `vit_emotion_model.pth`: Face model.
     - `emotion_model.keras`: Voice model.
     - `scaler.pkl`, `encoder.pkl`, `model_config.pkl`: Voice model configurations.

2. **Run the System**:
   ```bash
   python main.py
   ```

3. **Options**:
   - **Real-Time Emotion Detection**: Detect emotions from live webcam and microphone input.
   - **Sample Collection**: Capture image and audio samples for analysis.
   - **Visualization**: View logged predictions and analyze emotion distributions.

---

## Visualization and Logging

- **Emotion Distributions**:
  - Displays count plots for face, voice, and fused emotions.
  - Helps identify patterns in predictions.

- **Confidence Scores**:
  - Visualizes confidence distributions for face and voice models.

- **User Feedback**:
  - Allows users to provide feedback on fused predictions.
  - Updates the log file with feedback for analysis.

---

## Future Work

1. **Expand Datasets**: Train models on larger and more diverse datasets for improved performance.
2. **Fine-Tune Models**: Further optimize models for real-time performance.
3. **Custom GUI**: Develop a graphical user interface for easier interaction.
4. **Enhanced Fusion**: Experiment with advanced fusion techniques, such as weighted voting or neural networks.

---

Feel free to fork, contribute, or report issues. Happy coding! 🎉
