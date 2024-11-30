
# Multimodal Emotion Recognition System

This repository contains a **real-time multimodal emotion recognition system** that combines **facial expressions** and **vocal cues** to detect emotions accurately. By leveraging advanced models such as Vision Transformer (ViT) for facial analysis and BiLSTM for vocal analysis, the system achieves high performance by fusing predictions from both modalities.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Datasets](#datasets)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Training](#training)
  - [Face Emotion Recognition](#face-emotion-recognition)
  - [Vocal Emotion Recognition](#vocal-emotion-recognition)
- [Real-Time Multimodal Emotion Recognition](#real-time-multimodal-emotion-recognition)
- [Visualization and Logging](#visualization-and-logging)
- [Future Work](#future-work)
- [Authors and Contributions](#authors-and-contributions)

---

## Introduction

This project aims to bridge the gap between human emotions and machine interactions by creating a real-time **Multimodal Emotion Recognition System**. The system integrates two input modalities:

1. **Facial Expressions**: Captured via webcam and processed using a **Vision Transformer (ViT)** for emotion classification.
2. **Vocal Cues**: Captured via microphone and processed using a **Bidirectional Long Short-Term Memory (BiLSTM)** network for emotion detection.

These outputs are combined using a **decision-level fusion mechanism**, ensuring robust predictions even in challenging scenarios like noisy environments or poor lighting conditions.

---

## Features

- **Face Emotion Detection**: 
  - Uses OpenCV's SSD-based face detection (`res10_300x300_ssd_iter_140000.caffemodel`) and **Vision Transformer** for analyzing facial expressions.
- **Vocal Emotion Detection**: 
  - Utilizes a BiLSTM network trained on mel spectrogram and MFCC features for vocal emotion recognition.
- **Multimodal Fusion**: 
  - Combines predictions from both modalities using a decision-level fusion strategy.
- **Real-Time Processing**:
  - Processes live webcam and microphone input for real-time emotion detection.
- **Visualization and Feedback**:
  - Logs predictions for analysis and supports user feedback to refine the system.

---

## Datasets

This project uses publicly available datasets:

1. **FER2013** (Facial Emotion Recognition):  
   - Source: [FER2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
   - Description: Contains 35,887 grayscale images, each labeled with one of seven emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.

2. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song):  
   - Source: [RAVDESS Dataset on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
   - Description: Contains labeled speech audio samples for emotions like `angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, and `surprise`.

3. **TESS** (Toronto Emotional Speech Set):  
   - Source: [TESS Dataset on Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
   - Description: Includes 2,800 audio recordings, covering seven emotions recorded in noise-controlled environments.

---

## Technologies Used

- **Deep Learning Frameworks**: PyTorch, TensorFlow/Keras
- **Computer Vision**: OpenCV, Vision Transformer (ViT)
- **Audio Processing**: Librosa, PyAudio
- **Visualization**: Matplotlib, Seaborn
- **Data Handling**: NumPy, Pandas

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/multimodal-emotion-recognition.git
   cd multimodal-emotion-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the models and required files are in the `models/` directory:
   - `vit_emotion_model.pth`: Vision Transformer model for face emotion detection.
   - `emotion_model.keras`: BiLSTM model for voice emotion detection.
   - Supporting files: `scaler.pkl`, `encoder.pkl`, `model_config.pkl`, `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`.

---

## Training

### Face Emotion Recognition

1. **Model Architecture**: Vision Transformer (ViT) pretrained on ImageNet.
2. **Dataset**: FER2013
   - Preprocessing: Images resized to 224x224 and augmented using techniques like horizontal flips, rotations, and color jittering.
3. **Output**: Model saved as `vit_emotion_model.pth`.

### Vocal Emotion Recognition

1. **Model Architecture**: BiLSTM network with mel spectrogram and MFCC features.
2. **Datasets**: RAVDESS and TESS
   - Features combined into a 141-dimensional vector (128 mel bands + 13 MFCCs).
3. **Output**: Model saved as `emotion_model.keras`.

---

## Real-Time Multimodal Emotion Recognition

1. **Facial Input**:
   - Captured using OpenCV and preprocessed with SSD face detection (`res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt`).
   - Emotion prediction using Vision Transformer.

2. **Vocal Input**:
   - Captured via microphone, processed to extract mel spectrogram and MFCC features.
   - Emotion prediction using BiLSTM.

3. **Fusion**:
   - **Decision-Level Fusion**:
     - Agreement: Selects the emotion if both modalities agree.
     - Disagreement: Chooses the emotion with the higher confidence score.

4. **Output**:
   - Displays fused emotion with confidence levels in real-time.

---

## Visualization and Logging

- **Emotion Distribution**:
  - Generates plots of face, voice, and fused emotion distributions.
- **Confidence Scores**:
  - Visualizes confidence scores for both modalities.
- **User Feedback**:
  - Logs user feedback to refine the system.

---

## Future Work

1. **Expand Datasets**:
   - Include more diverse datasets to improve model robustness.
2. **Optimize Models**:
   - Use techniques like quantization and pruning for faster processing.
3. **Advanced Fusion**:
   - Experiment with dynamic and attention-based fusion methods.
4. **Real-Time Feedback**:
   - Integrate live user feedback for system refinement.

---

## Authors and Contributions

- **Mostafa Rafiur Wasib**: Developed Vision Transformer model for facial emotion recognition and handled dataset preprocessing.
- **Chowdhury Nafis Saleh**: Designed BiLSTM network for vocal emotion recognition and feature extraction.
- **Azmol Fuad**: Developed fusion mechanism and real-time interface integration.

---

Feel free to fork, contribute, or report issues. Happy coding! 🎉
