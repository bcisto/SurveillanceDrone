import time
import math
from picarx import Picarx
from face_scanner import FaceScanner
from movement_control import MovementControl

class AutonomousDrive:
    def __init__(self):
        # Initialize PiCar-X
        self.px = Picarx()
        
        # Initialize movement control
        self.movement = MovementControl(picarx_instance=self.px)
        
        # Constants for movement
        self.SPEED = 50
        
        # Distance thresholds (in cm)
        self.SAFE_DISTANCE = 40     # > 40 safe
        self.DANGER_DISTANCE = 20   # > 20 && < 40 turn around, < 20 backward
        
        # Movement tracking
        self.distance_traveled = 0   # Distance traveled in inches
        self.target_distance = 60    # 5 feet = 60 inches
        self.last_move_time = time.time()
        
        # Initialize face scanner with our Picarx instance
        self.face_scanner = FaceScanner(picarx_instance=self.px)
        
    def get_distance(self):
        """Get distance reading from ultrasonic sensor."""
        distance = self.px.ultrasonic.read()
        if 0 < distance < 300:  # Filter invalid readings
            return distance
        return 300  # Return max distance if invalid reading
        
    def update_distance_traveled(self):
        """Update the distance traveled based on time and speed."""
        current_time = time.time()
        elapsed_time = current_time - self.last_move_time
        
        # Approximate distance traveled (speed * time)
        # At speed 50, car moves roughly 20 inches per second
        distance_increment = (self.SPEED / 50) * 20 * elapsed_time
        self.distance_traveled += distance_increment
        self.last_move_time = current_time
        
    def handle_face_detection(self, face_info):
        """Handle face detection by stopping and turning."""
        face_x, face_y, face_w, face_h = face_info
        print(f"\nFace detected at ({face_x}, {face_y})! Stopping...")
        
        # Stop the car
        self.movement.stop()
        time.sleep(1)  # Pause for a second
        
        # Turn away from the face based on its position
        turn_angle = 30 if face_x < 320 else -30  # Turn right if face is on left, left if face is on right
        self.movement.smooth_turn(turn_angle, duration=0.5)
        
        print("Resuming forward movement...")
        
    def drive_forward(self):
        """Drive forward while avoiding obstacles and detecting faces."""
        print("Starting autonomous drive - Target: 5 feet forward")
        
        # Start the face scanner
        self.face_scanner.start_camera()
        
        try:
            while self.distance_traveled < self.target_distance:
                # Update camera pan
                self.face_scanner.update_camera_pan()
                
                # Check for faces
                face_detected, face_info = self.face_scanner.check_for_faces()
                if face_detected:
                    self.handle_face_detection(face_info)
                    continue
                
                # Get distance reading
                distance = self.get_distance()
                print(f"\rDistance: {distance:.1f}cm, Traveled: {self.distance_traveled:.1f} inches, Camera: {self.face_scanner.camera_angle:.1f}°", end='')
                
                if distance >= self.SAFE_DISTANCE:
                    # Safe to move forward
                    self.movement.move_forward(self.SPEED)
                    self.update_distance_traveled()
                    
                elif distance >= self.DANGER_DISTANCE:
                    # Warning zone - turn around
                    self.movement.turn_right()
                    time.sleep(0.1)
                    
                else:
                    # Danger zone - back up
                    self.movement.turn_left()
                    self.movement.move_backward(self.SPEED)
                    time.sleep(0.5)
                    # Subtract from distance traveled
                    self.distance_traveled = max(0, self.distance_traveled - 10)
                
                time.sleep(0.02)  # Short delay for smooth operation
                
            # Reached target distance
            print("\nReached target distance of 5 feet!")
            self.movement.stop()
            
        except KeyboardInterrupt:
            print("\nDrive interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources and reset car state."""
        print("Cleaning up...")
        self.movement.cleanup()
        self.face_scanner.cleanup()

def main():
    driver = AutonomousDrive()
    try:
        driver.drive_forward()
    except Exception as e:
        print(f"Error during autonomous driving: {e}")
    finally:
        driver.cleanup()

if __name__ == '__main__':
    main() 