import time
import os
import threading
import cv2
import numpy as np
from picamera2 import Picamera2
import picarx
import robot_hat

data_dir = 'face_cards/'
route_waypoints = [
    {'x': 0, 'y': 0},
    {'x': 1, 'y': 0},
    {'x': 1, 'y': 1},
    {'x': 0, 'y': 1},
]
speed = 30
frame_width = 320
frame_height = 240
recognizer_threshold = 70.0
auth_labels = {}
stop_event = threading.Event()

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

def alert_intruder(frame, location, intruder_dir='intruders/'):
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs(intruder_dir, exist_ok=True)
    path = os.path.join(intruder_dir, f'intruder_{timestamp}.jpg')
    cv2.imwrite(path, frame)
    print(f'[ALERT] Intruder at {location}, saved to {path}')
    robot_hat.buzzer.on(); time.sleep(1); robot_hat.buzzer.off()

def drive_route(picar):
    try:
        while not stop_event.is_set():
            for wp in route_waypoints:
                if stop_event.is_set(): break
                print(f'[MOVE] Heading to {wp}')
                time.sleep(5)
    finally:
        pass


def scan_loop(picam2, picar):
    try:
        while not stop_event.is_set():
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
                    alert_intruder(frame, None)
            time.sleep(0.1)
    finally:
        pass

def main():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={'format':'RGB888','size':(frame_width,frame_height)})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    picar = picarx.Picarx()
    load_and_train(data_dir)
    drive_thread = threading.Thread(target=drive_route, args=(picar,), daemon=True)
    scan_thread = threading.Thread(target=scan_loop, args=(picam2, picar), daemon=True)
    drive_thread.start()
    scan_thread.start()
    try:
        while drive_thread.is_alive() and scan_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print('[INFO] Stopping patrol')
        stop_event.set()
    finally:
        drive_thread.join()
        scan_thread.join()
        picar.stop()
        picam2.stop()
        print('[INFO] Patrol ended')

if __name__ == '__main__':
    main()