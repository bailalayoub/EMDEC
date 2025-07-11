import cv2
import numpy as np
from flask import Flask, Response, render_template
from tensorflow.keras.models import load_model

EMOTIONS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral'
}

app = Flask(__name__)
model = None
cap = None


def gen_frames():
    global cap, model
    while True:
        success, frame = cap.read()
        if not success:
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def create_app(model_path):
    global model, cap
    model = load_model(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Unable to open camera')
    return app


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Flask emotion detection app')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    create_app(args.model)
    app.run(host=args.host, port=args.port, debug=False)
