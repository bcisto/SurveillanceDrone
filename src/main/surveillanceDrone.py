import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2
import robot_hat

data_dir = 'face_cards/'
frame_width = 320
frame_height = 240
recognizer_threshold = 70.0
auth_labels = {}

def init_face_tools():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")
    detector = cv2.CascadeClassifier(cascade_path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    return detector, recognizer

face_detector, face_recognizer = init_face_tools()

def load_and_train(data_dir):
    samples, labels = [], []
    next_label = 0
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        id_name = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(data_dir, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            print(f'[WARN] No face in {fname}, skipping.')
            continue
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        label = next_label
        auth_labels[label] = id_name
        samples.append(roi)
        labels.append(label)
        print(f'[INFO] Added {id_name} as label {label}')
        next_label += 1
    if not samples:
        raise RuntimeError('No valid face cards found')
    face_recognizer.train(samples, np.array(labels))
    print('[INFO] Face recognizer trained')

def alert_intruder(frame, location=None, intruder_dir='intruders/'):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(intruder_dir, exist_ok=True)
    path = os.path.join(intruder_dir, f'intruder_{timestamp}.jpg')
    cv2.imwrite(path, frame)
    print(f'[ALERT] Intruder detected, saved to {path}')
    robot_hat.buzzer.on(); time.sleep(1); robot_hat.buzzer.off()

def scan_loop(picam2):
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                label, confidence = face_recognizer.predict(roi)
                if confidence < recognizer_threshold:
                    name = auth_labels.get(label, 'Unknown')
                    print(f'[INFO] Authorized: {name} ({confidence:.1f})')
                else:
                    print(f'[WARN] Unauthorized: confidence={confidence:.1f}')
                    alert_intruder(frame)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('[INFO] Scan stopped by user')

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={'format': 'RGB888', 'size': (frame_width, frame_height)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    load_and_train(data_dir)
    print('[INFO] Starting face scan loop')
    scan_loop(picam2)
    picam2.stop()
    print('[INFO] Patrol ended')

if __name__ == '__main__':
    main()