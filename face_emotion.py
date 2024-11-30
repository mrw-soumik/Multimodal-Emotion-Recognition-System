# face_emotion.py

import os
import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from timm import create_model
from PIL import Image
from config import DEVICE, FACE_EMOTIONS, PROTOTXT_PATH, FACE_DETECTOR_PATH

face_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])

class ViTEmotionModel(nn.Module):

    def __init__(self, num_classes):
        super(ViTEmotionModel, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class FaceEmotionDetector:

    def __init__(self, model_path):
        self.model = ViTEmotionModel(num_classes=len(FACE_EMOTIONS)).to(DEVICE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face model file not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print("Face emotion detection model loaded successfully.")

        try:
            self.face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, FACE_DETECTOR_PATH)
            print("Face detector models loaded successfully.")
        except Exception as e:
            print(f"Error loading face detector models: {e}")
            self.face_net = None

    def detect_faces(self, frame):
        if self.face_net is None:
            return []

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     scalefactor=1.0,
                                     size=(300, 300),
                                     mean=(104.0, 177.0, 123.0))
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x2, y2) = box.astype("int")
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                face = frame[y:y2, x:x2]
                faces.append((face, (x, y, x2, y2)))
        return faces

    def predict_emotion(self, face):
        try:
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = face_transform(face_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = self.model(face_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                emotion = FACE_EMOTIONS[pred_idx]
                confidence_score = torch.nn.functional.softmax(output, dim=1)[
                    0, pred_idx].item() * 100
                return emotion, confidence_score
        except Exception as e:
            print(f"Error during face emotion prediction: {e}")
            return None, None

    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        results = []
        for face, box in faces:
            emotion, confidence = self.predict_emotion(face)
            if emotion:
                results.append((emotion, confidence, box))
        return results

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, None
        faces = self.detect_faces(image)
        if faces:
            face, _ = faces[0]
            return self.predict_emotion(face)
        else:
            print("No face detected in the image.")
            return None, None
