import time
import math
from picarx import Picarx

class MovementControl:
    def __init__(self, picarx_instance=None):
        """Initialize movement control with optional Picarx instance."""
        # Use provided Picarx instance or create new one
        self.px = picarx_instance if picarx_instance else Picarx()
        
        # Movement settings
        self.DEFAULT_SPEED = 50
        self.TURN_SPEED = 30
        self.DEFAULT_TURN_ANGLE = 30
        
        # Initialize position
        self.px.set_dir_servo_angle(0)
        time.sleep(0.1)  # Give time for servo to settle
        
    def move_forward(self, speed=None):
        """Move forward at specified speed."""
        speed = speed if speed is not None else self.DEFAULT_SPEED
        self.px.set_dir_servo_angle(0)
        self.px.forward(speed)
        
    def move_backward(self, speed=None):
        """Move backward at specified speed."""
        speed = speed if speed is not None else self.DEFAULT_SPEED
        self.px.set_dir_servo_angle(0)
        self.px.backward(speed)
        
    def turn_left(self, angle=None, speed=None):
        """Turn left at specified angle and speed."""
        angle = abs(angle) if angle is not None else self.DEFAULT_TURN_ANGLE
        speed = speed if speed is not None else self.TURN_SPEED
        self.px.set_dir_servo_angle(-angle)
        self.px.forward(speed)
        
    def turn_right(self, angle=None, speed=None):
        """Turn right at specified angle and speed."""
        angle = abs(angle) if angle is not None else self.DEFAULT_TURN_ANGLE
        speed = speed if speed is not None else self.TURN_SPEED
        self.px.set_dir_servo_angle(angle)
        self.px.forward(speed)
        
    def stop(self):
        """Stop all movement."""
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)
        
    def turn_to_angle(self, target_angle, speed=None):
        """Turn to a specific angle (positive = right, negative = left)."""
        speed = speed if speed is not None else self.TURN_SPEED
        # Ensure angle is within valid range
        target_angle = max(min(target_angle, 40), -40)
        
        # Set steering angle
        self.px.set_dir_servo_angle(target_angle)
        self.px.forward(speed)
        
    def parallel_park_left(self):
        """Execute a left parallel parking maneuver."""
        # Initial turn
        self.turn_left(angle=40)
        time.sleep(0.5)
        
        # Backward into spot
        self.px.set_dir_servo_angle(40)
        self.px.backward(self.DEFAULT_SPEED)
        time.sleep(1.0)
        
        # Straighten out
        self.px.set_dir_servo_angle(-20)
        self.px.backward(self.DEFAULT_SPEED)
        time.sleep(0.5)
        
        # Final adjustment
        self.stop()
        
    def parallel_park_right(self):
        """Execute a right parallel parking maneuver."""
        # Initial turn
        self.turn_right(angle=40)
        time.sleep(0.5)
        
        # Backward into spot
        self.px.set_dir_servo_angle(-40)
        self.px.backward(self.DEFAULT_SPEED)
        time.sleep(1.0)
        
        # Straighten out
        self.px.set_dir_servo_angle(20)
        self.px.backward(self.DEFAULT_SPEED)
        time.sleep(0.5)
        
        # Final adjustment
        self.stop()
        
    def three_point_turn(self):
        """Execute a three-point turn."""
        # First part - turn right forward
        self.turn_right(angle=40)
        time.sleep(1.0)
        
        # Second part - turn left backward
        self.px.set_dir_servo_angle(-40)
        self.px.backward(self.DEFAULT_SPEED)
        time.sleep(1.0)
        
        # Third part - turn right forward to straighten
        self.turn_right(angle=20)
        time.sleep(0.5)
        
        # Final adjustment
        self.stop()
        
    def smooth_turn(self, target_angle, duration=1.0):
        """Execute a smooth turn over specified duration."""
        start_time = time.time()
        while time.time() - start_time < duration:
            # Calculate progress (0 to 1)
            progress = (time.time() - start_time) / duration
            # Use sine function for smooth acceleration and deceleration
            smoothed_progress = math.sin(progress * math.pi / 2)
            # Calculate current angle
            current_angle = target_angle * smoothed_progress
            # Apply angle
            self.turn_to_angle(current_angle)
            time.sleep(0.02)
        self.stop()
        
    def cleanup(self):
        """Clean up and reset movement."""
        self.stop()
        time.sleep(0.1) 