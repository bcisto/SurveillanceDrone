import os
import time
import cv2
from picamera2 import Picamera2

DATA_DIR = 'face_cards'
os.makedirs(DATA_DIR, exist_ok=True)

def capture_face_card():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    window_name = "Capture Face Card"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.startWindowThread()
    print("Click the window to focus, then press SPACE to capture, or ESC to cancel.")
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, frame)

            k = cv2.waitKey(1) & 0xFF
            if k != 0xFF:
                print(f"[DEBUG] key code = {k}")
            if k == 27:
                print("Capture cancelled.")
                break
            elif k == ord(' '):
                person_id = input("Enter new person ID (no spaces): ").strip()
                if not person_id:
                    print("Invalid ID, try again.")
                    continue
                fname = os.path.join(DATA_DIR, f"{person_id}.jpg")
                cv2.imwrite(fname, frame)
                print(f"Saved face card: {fname}")
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    capture_face_card()