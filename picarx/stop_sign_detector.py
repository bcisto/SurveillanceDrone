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
        
        # Initialize Vilib camera and detection
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)
        time.sleep(0.8)  # Wait for camera to initialize
        
        # Configure object detection
        Vilib.detect_obj_parameter['threshold'] = confidence_threshold
        Vilib.detect_obj_parameter['target_n'] = 1  # Number of target objects
        Vilib.object_detect_switch(True)  # Enable object detection
        
    def track_stop_sign(self):
        """Track stop sign with camera if detected."""
        if Vilib.detect_obj_parameter.get('object_n', 0) > 0:
            # Get coordinates of detected object
            x = Vilib.detect_obj_parameter.get('object_x', 0)
            y = Vilib.detect_obj_parameter.get('object_y', 0)
            
            # Update camera angles to track object
            self.x_angle += (x * 10 / 640) - 5
            self.x_angle = clamp_number(self.x_angle, -35, 35)
            self.px.set_cam_pan_angle(self.x_angle)
            
            self.y_angle -= (y * 10 / 480) - 5
            self.y_angle = clamp_number(self.y_angle, -35, 35)
            self.px.set_cam_tilt_angle(self.y_angle)
            
            return True
        return False
    
    def get_stop_sign_info(self):
        """Get current stop sign detection information."""
        if Vilib.detect_obj_parameter.get('object_n', 0) > 0:
            return {
                'detected': True,
                'x': Vilib.detect_obj_parameter.get('object_x', 0),
                'y': Vilib.detect_obj_parameter.get('object_y', 0),
                'width': Vilib.detect_obj_parameter.get('object_w', 0),
                'height': Vilib.detect_obj_parameter.get('object_h', 0)
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