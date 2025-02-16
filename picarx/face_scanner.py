from vilib import Vilib
import time
import signal
import sys
import math
from picarx import Picarx

class FaceScanner:
    def __init__(self, picarx_instance=None):
        """Initialize the face scanner with camera settings."""
        # Initialize camera
        self.camera_started = False
        self.scanning = True
        
        # Use provided Picarx instance or create new one
        self.px = picarx_instance if picarx_instance else Picarx()
        
        # Camera panning settings
        self.camera_angle = 0
        self.PAN_RANGE = 35         # Maximum pan angle (left/right)
        self.BASE_SPEED = 10        # Base speed in degrees per second
        self.current_speed = 0      # Current panning speed
        self.MAX_SPEED = 15         # Maximum speed in degrees per second
        self.ACCELERATION = 20      # Degrees per second squared
        self.panning_right = True   # Direction flag
        self.last_pan_time = time.time()
        self.last_print_time = time.time()  # For controlling status message rate
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def check_for_faces(self):
        """Check if any faces are detected and return face info."""
        if not self.camera_started:
            return False, None
            
        if Vilib.detect_obj_parameter['human_n'] != 0:
            # Get face detection data
            face_x = Vilib.detect_obj_parameter['human_x']
            face_y = Vilib.detect_obj_parameter['human_y']
            face_w = Vilib.detect_obj_parameter['human_w']
            face_h = Vilib.detect_obj_parameter['human_h']
            return True, (face_x, face_y, face_w, face_h)
        return False, None
        
    def smooth_acceleration(self, current_angle):
        """Calculate smooth acceleration/deceleration based on position."""
        # Calculate distance from edges
        distance_from_edge = min(
            abs(self.PAN_RANGE - abs(current_angle)),  # Distance to max angle
            5  # Start slowing down 5 degrees from edge
        )
        
        # Smooth acceleration/deceleration
        if distance_from_edge < 5:
            # Decelerate near edges
            target_speed = self.BASE_SPEED * (distance_from_edge / 5)
        else:
            # Normal speed in middle range
            target_speed = self.MAX_SPEED
            
        return target_speed
        
    def update_camera_pan(self):
        """Smoothly pan camera left and right with fluid motion."""
        current_time = time.time()
        elapsed_time = current_time - self.last_pan_time
        
        # Get target speed based on position
        target_speed = self.smooth_acceleration(self.camera_angle)
        
        # Smoothly adjust current speed towards target
        speed_change = self.ACCELERATION * elapsed_time
        if self.current_speed < target_speed:
            self.current_speed = min(self.current_speed + speed_change, target_speed)
        else:
            self.current_speed = max(self.current_speed - speed_change, target_speed)
        
        # Calculate angle change with current speed
        angle_change = self.current_speed * elapsed_time
        
        # Update angle with direction
        if self.panning_right:
            self.camera_angle += angle_change
            if self.camera_angle >= self.PAN_RANGE:
                print("\nReaching right limit, changing direction...")  # Debug info
                self.camera_angle = self.PAN_RANGE
                self.panning_right = False
                self.current_speed = self.BASE_SPEED  # Start with base speed instead of 0
        else:
            self.camera_angle -= angle_change
            if self.camera_angle <= -self.PAN_RANGE:
                print("\nReaching left limit, changing direction...")  # Debug info
                self.camera_angle = -self.PAN_RANGE
                self.panning_right = True
                self.current_speed = self.BASE_SPEED  # Start with base speed instead of 0
                
        # Update camera position - use float for smoother motion
        try:
            self.px.set_cam_pan_angle(self.camera_angle)
            # Print debug info occasionally
            if current_time - self.last_print_time >= 0.2:
                print(f"\rSpeed: {self.current_speed:.1f}°/s, Target: {target_speed:.1f}°/s", end='')
        except Exception as e:
            print(f"\nError setting camera angle: {e}")
            
        self.last_pan_time = current_time
        
    def start_camera(self):
        """Start the camera and initialize face detection."""
        try:
            print("Starting camera...")
            Vilib.camera_start(vflip=False, hflip=False)
            Vilib.display(local=True, web=True)  # Enable both local and web display
            Vilib.face_detect_switch(True)  # Enable face detection
            self.camera_started = True
            
            # Initialize camera position
            self.px.set_cam_pan_angle(0)
            time.sleep(0.1)  # Give time for servo to settle
            
            print("Camera started successfully!")
            print("Face detection enabled - press Ctrl+C to exit")
            print("\nScanning area for faces...")
            print("\nDebug info: Showing current speed and target speed")
            
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.cleanup()
            sys.exit(1)
            
    def scan_faces(self):
        """Main loop for face detection."""
        try:
            while self.scanning:
                # Update camera position
                self.update_camera_pan()
                
                # Update status message at a reasonable rate (every 0.2 seconds)
                current_time = time.time()
                if current_time - self.last_print_time >= 0.2:
                    direction = "→" if self.panning_right else "←"
                    if Vilib.detect_obj_parameter['human_n'] != 0:
                        # Get face detection data
                        face_x = Vilib.detect_obj_parameter['human_x']
                        face_y = Vilib.detect_obj_parameter['human_y']
                        face_w = Vilib.detect_obj_parameter['human_w']
                        face_h = Vilib.detect_obj_parameter['human_h']
                        print(f"\rScanning {direction} | Face detected at: ({face_x}, {face_y}), Size: {face_w}x{face_h} | Angle: {self.camera_angle:.1f}°", end='')
                    else:
                        print(f"\rScanning {direction} | No faces detected | Angle: {self.camera_angle:.1f}°", end='')
                    self.last_print_time = current_time
                    
                time.sleep(0.02)  # Smaller delay for even smoother motion
                
        except Exception as e:
            print(f"\nError during face scanning: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        if self.camera_started:
            print("\nClosing camera...")
            Vilib.face_detect_switch(False)
            Vilib.camera_close()
            # Smoothly return to center
            while abs(self.camera_angle) > 0.5:
                self.camera_angle *= 0.8
                self.px.set_cam_pan_angle(self.camera_angle)
                time.sleep(0.02)
            self.px.set_cam_pan_angle(0)
            self.camera_started = False
            
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\nStopping face scanner...")
        self.scanning = False

def main():
    scanner = FaceScanner()
    scanner.start_camera()
    scanner.scan_faces()

if __name__ == '__main__':
    main() 