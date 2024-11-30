# multimodal.py

def fuse_emotions(face_emotion, face_confidence, voice_emotion, voice_confidence):
    if face_emotion == 'Unknown' or face_confidence is None:
        face_confidence = 0.0
    if voice_emotion == 'Unknown' or voice_confidence is None:
        voice_confidence = 0.0

    if face_emotion != 'Unknown' and voice_emotion != 'Unknown':
        if face_emotion == voice_emotion:
            return face_emotion
        else:
            if face_confidence > voice_confidence:
                return face_emotion
            else:
                return voice_emotion
    elif face_emotion != 'Unknown':
        return face_emotion
    elif voice_emotion != 'Unknown':
        return voice_emotion
    else:
        return 'Unknown'
