import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

def main(args):
    model = load_model(args.model)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Unable to access the camera')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        rgb = rgb / 255.0
        rgb = np.expand_dims(rgb, axis=0)
        preds = model.predict(rgb)
        label = EMOTIONS[int(np.argmax(preds))]
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run emotion detection on webcam')
    parser.add_argument('--model', required=True, help='Path to trained model (.h5)')
    main(parser.parse_args())
