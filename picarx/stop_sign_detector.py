import argparse
import sys
import time
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from vilib import Vilib
from collections import deque
import threading

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
INFERENCE_SIZE = (320, 320)
MAX_NUM_THREADS = 3  # Use 3 threads to leave 1 core free for system tasks

# COCO dataset class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

class ObjectDetector:
    def __init__(self, model_path, num_threads=MAX_NUM_THREADS, score_threshold=0.5):
        """Initialize detector with Pi 4 optimized settings."""
        # Initialize TFLite interpreter with XNNPACK delegate
        self.interpreter = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.score_threshold = score_threshold
        
        # Pre-calculate colors for each class
        self.colors = {class_name: (hash(class_name) % 256, 
                                  hash(class_name * 2) % 256, 
                                  hash(class_name * 3) % 256) 
                      for class_name in COCO_CLASSES}
        
        # Performance tracking with smaller window for more responsive FPS display
        self.fps_deque = deque(maxlen=10)  # Track last 10 frames for FPS
        self.frame_times = deque(maxlen=10)  # Track processing times
        
    def preprocess_image(self, frame):
        """Optimized image preprocessing for Pi 4."""
        # Use INTER_AREA for downscaling (better quality for significant size reduction)
        img = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.expand_dims(img.astype(np.uint8), axis=0)
    
    def detect_objects(self, frame):
        """Run detection on a frame."""
        # Time the processing
        start_time = time.time()
        
        # Preprocess
        input_data = self.preprocess_image(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Track processing time
        process_time = time.time() - start_time
        self.frame_times.append(process_time)
        
        return boxes, classes, scores, process_time
    
    def draw_detections(self, frame, boxes, classes, scores):
        """Draw detection results efficiently."""
        detected = set()
        height, width = frame.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] >= self.score_threshold:
                class_id = int(classes[i])
                if class_id < len(COCO_CLASSES):
                    class_name = COCO_CLASSES[class_id]
                    detected.add(class_name)
                    
                    # Convert coordinates
                    ymin, xmin, ymax, xmax = boxes[i]
                    xmin, xmax = int(xmin * width), int(xmax * width)
                    ymin, ymax = int(ymin * height), int(ymax * height)
                    
                    # Draw box and label
                    color = self.colors[class_name]
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    
                    label = f"{class_name}: {scores[i]:.2f}"
                    cv2.putText(frame, label, (xmin, ymin - 10),
                              cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        
        # Calculate and show performance metrics
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (24, 20),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Process Time: {avg_time*1000:.0f}ms", (24, 40),
                       cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        
        return detected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path of the object detection model.',
                       default='models/efficientdet_lite0.tflite')
    parser.add_argument('--numThreads', help='Number of CPU threads to run the model.',
                       type=int, default=MAX_NUM_THREADS)
    parser.add_argument('--scoreThreshold', help='Detection score threshold',
                       type=float, default=0.4)  # Slightly lower threshold for better detection
    args = parser.parse_args()

    print(f"Starting object detection optimized for Raspberry Pi 4B...")
    print(f"Using {INFERENCE_SIZE} inference size")
    print(f"Using {args.numThreads} threads")
    print(f"Detection threshold: {args.scoreThreshold}")
    
    print("Starting camera...")
    Vilib.camera_start()
    Vilib.display()
    
    print("Loading model...")
    detector = ObjectDetector(args.model, args.numThreads, args.scoreThreshold)
    print("Model loaded successfully!")
    print("Looking for objects... (Press 'q' to quit)")
    
    detected_objects = set()
    running = True
    
    try:
        while running:
            try:
                frame = Vilib.img.copy()
                if frame is None:
                    continue
                
                # Run detection
                boxes, classes, scores, process_time = detector.detect_objects(frame)
                
                # Draw results and get detected objects
                current_detections = detector.draw_detections(frame, boxes, classes, scores)
                
                # Report changes in detections
                new_detections = current_detections - detected_objects
                lost_detections = detected_objects - current_detections
                
                if new_detections:
                    print(f"New objects detected: {', '.join(new_detections)}")
                if lost_detections:
                    print(f"Objects lost from view: {', '.join(lost_detections)}")
                
                detected_objects = current_detections
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except KeyboardInterrupt:
                print("\nShutting down gracefully...")
                running = False
            except Exception as e:
                print(f"Error during detection: {e}")
                running = False
    
    finally:
        try:
            print("Closing camera...")
            Vilib.camera_close()
            cv2.destroyAllWindows()
            print("Shutdown complete")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        print("Exiting...") 