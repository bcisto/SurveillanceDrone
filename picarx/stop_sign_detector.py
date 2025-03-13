from vilib import Vilib
from picarx import Picarx
import time

def clamp_number(num, a, b):
    return max(min(num, max(a, b)), min(a, b))

class StopSignDetector:
    def __init__(self, confidence_threshold=0.5):
        """Initialize the stop sign detector using Vilib."""
        self.confidence_threshold = confidence_threshold
        self.px = Picarx()
        self.x_angle = 0
        self.y_angle = 0
        
        # Test wheel movement at startup
        print("Testing wheel movement...")
        self.px.set_dir_servo_angle(30)
        time.sleep(1)
        self.px.set_dir_servo_angle(-30)
        time.sleep(1)
        self.px.set_dir_servo_angle(0)
        print("Wheel test complete")
        
        # Initialize Vilib camera and detection
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)
        time.sleep(0.8)  # Wait for camera to initialize
        
        # Configure object detection
        Vilib.detect_obj_parameter['threshold'] = confidence_threshold
        Vilib.detect_obj_parameter['target_n'] = 1
        Vilib.object_detect_switch(True)
        print("Object detection configured")
        
    def track_stop_sign(self):
        """Track stop sign with camera if detected and turn wheels."""
        try:
            # Check if any object is detected
            detected = False
            if Vilib.detect_obj_parameter.get('object_n', 0) > 0:
                # Check if the detected object is a stop sign
                class_name = Vilib.detect_obj_parameter.get('class_name', '')
                confidence = Vilib.detect_obj_parameter.get('confidence', 0)
                print(f"Detected object class: {class_name}, confidence: {confidence}")
                
                if 'stop' in class_name.lower():
                    detected = True
                    x = Vilib.detect_obj_parameter.get('object_x', 0)
                    y = Vilib.detect_obj_parameter.get('object_y', 0)
                    print(f"Stop sign location: x={x}, y={y}")
                    
                    # Force wheel movement for testing
                    print("Forcing wheel movement test...")
                    print("Turning wheels right")
                    self.px.set_dir_servo_angle(30)
                    time.sleep(0.5)
                    print("Turning wheels left")
                    self.px.set_dir_servo_angle(-30)
                    time.sleep(0.5)
                    print("Centering wheels")
                    self.px.set_dir_servo_angle(0)
                    
                    return True
            
            if not detected:
                print("No stop sign detected in this frame")
            return False
                
        except Exception as e:
            print(f"Error in track_stop_sign: {e}")
            return False
            
    def get_stop_sign_info(self):
        """Get current stop sign detection information."""
        if Vilib.detect_obj_parameter.get('object_n', 0) > 0:
            class_name = Vilib.detect_obj_parameter.get('class_name', '')
            if 'stop' in class_name.lower():
                return {
                    'detected': True,
                    'x': Vilib.detect_obj_parameter.get('object_x', 0),
                    'y': Vilib.detect_obj_parameter.get('object_y', 0),
                    'width': Vilib.detect_obj_parameter.get('object_w', 0),
                    'height': Vilib.detect_obj_parameter.get('object_h', 0),
                    'confidence': Vilib.detect_obj_parameter.get('confidence', 0)
                }
        return {'detected': False}
    
    def cleanup(self):
        """Clean up resources."""
        self.px.stop()
        Vilib.object_detect_switch(False)  # Disable object detection
        Vilib.camera_close()

def main():
    print("Starting stop sign detection using Vilib...")
    
    # Initialize detector
    detector = StopSignDetector(confidence_threshold=0.5)
    print("Detection started! Press Ctrl+C to quit.")
    print("Camera will track any detected stop signs")
    
    try:
        while True:
            # Track stop sign if detected
            if detector.track_stop_sign():
                info = detector.get_stop_sign_info()
                print(f"\nStop Sign Detected!")
                print(f"  Position: ({info['x']}, {info['y']})")
                print(f"  Size: {info['width']}x{info['height']}")
            
            time.sleep(0.05)  # Small delay to prevent CPU overuse
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        print("\nCleaning up...")
        detector.cleanup()
        print("Shutdown complete")

if __name__ == '__main__':
    main() 