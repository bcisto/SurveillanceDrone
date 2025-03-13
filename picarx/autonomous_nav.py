from picarx import Picarx
import time
import numpy as np
import cv2
from vilib import Vilib
import tflite_runtime.interpreter as tflite
import os
from typing import Tuple, List, Dict
import math
import heapq

class AutonomousNavigator:
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize autonomous navigation with object detection and mapping."""
        # Initialize PiCar-X
        self.px = Picarx()
        
        # Initialize camera and detection
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)
        time.sleep(0.8)  # Wait for camera to initialize
        
        # Configure object detection
        Vilib.detect_obj_parameter['threshold'] = confidence_threshold
        Vilib.detect_obj_parameter['target_n'] = 1
        Vilib.object_detect_switch(True)
        
        # Load TensorFlow model for additional object detection
        self.interpreter = self.load_object_detector()
        if self.interpreter:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
        
        # Constants for movement
        self.SPEED = 25  # Reduced speed for better control
        self.TURN_SPEED = 15
        self.SCAN_STEP = 5
        
        # Distance thresholds (in cm)
        self.SAFE_DISTANCE = 40
        self.DANGER_DISTANCE = 20
        self.TURN_DISTANCE = 30
        
        # Initialize mapping grid (1 cell = 1 cm)
        self.grid_size = 200
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.car_pos = np.array([self.grid_size//2, 20])  # Start near bottom
        self.car_angle = 0  # degrees, 0 is forward
        
        # Navigation parameters
        self.goal_pos = None
        self.current_path = []
        self.path_index = 0
        self.GOAL_THRESHOLD = 15  # cm, distance to consider goal reached
        self.REPLAN_THRESHOLD = 30  # cm, distance to trigger path replanning
        
        # Camera movement
        self.camera_angle = 0
        self.MAX_CAM_ANGLE = 90
        
    def load_object_detector(self) -> tflite.Interpreter:
        """Load TensorFlow Lite model for object detection."""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'efficientdet_lite0.tflite')
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            print("Successfully loaded object detection model")
            return interpreter
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
            
    def detect_stop_sign(self) -> Tuple[bool, Dict]:
        """Detect stop signs using Vilib."""
        if Vilib.detect_obj_parameter.get('object_n', 0) > 0:
            class_name = Vilib.detect_obj_parameter.get('class_name', '')
            if 'stop' in class_name.lower():
                return True, {
                    'x': Vilib.detect_obj_parameter.get('object_x', 0),
                    'y': Vilib.detect_obj_parameter.get('object_y', 0),
                    'width': Vilib.detect_obj_parameter.get('object_w', 0),
                    'height': Vilib.detect_obj_parameter.get('object_h', 0),
                    'confidence': Vilib.detect_obj_parameter.get('confidence', 0)
                }
        return False, {}
        
    def scan_environment(self) -> List[Tuple[float, float]]:
        """Scan environment with ultrasonic sensor and camera pan."""
        readings = []
        scan_speed = 0.01  # Reduced from 0.02
        
        # Reduce scan range and increase step size for faster scanning
        for angle in range(-self.MAX_CAM_ANGLE, self.MAX_CAM_ANGLE + 1, self.SCAN_STEP * 2):
            self.px.set_cam_pan_angle(angle)
            time.sleep(scan_speed)
            
            # Single reading with validation instead of multiple readings
            distance = self.px.ultrasonic.read()
            if distance is not None and 0 < distance < 300:
                readings.append((distance, angle))
                # Update grid map
                self.update_grid(distance, angle)
                
            # Check for stop signs during scan
            stop_detected, stop_info = self.detect_stop_sign()
            if stop_detected:
                print(f"\nStop sign detected at angle {angle}°!")
                self.handle_stop_sign(stop_info)
        
        return readings
        
    def update_grid(self, distance: float, angle: float) -> None:
        """Update grid map with new obstacle reading."""
        if distance <= 0:
            return
            
        # Convert polar coordinates to grid coordinates
        angle_rad = np.radians(angle + self.car_angle)
        dx = distance * np.cos(angle_rad)
        dy = distance * np.sin(angle_rad)
        
        # Calculate obstacle position in grid
        obstacle_pos = self.car_pos + np.array([dx, dy])
        x, y = int(obstacle_pos[0]), int(obstacle_pos[1])
        
        # Mark obstacle if within bounds
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1
            
    def handle_stop_sign(self, stop_info: Dict) -> None:
        """Handle detected stop sign."""
        print("Handling stop sign...")
        print(f"Stop sign details: {stop_info}")
        
        # Stop the car
        self.px.stop()
        time.sleep(2)  # Wait at stop sign
        
        # Calculate relative position and decide how to proceed
        center_x = 320  # Assuming camera resolution is 640x480
        if stop_info['x'] < center_x - 50:
            print("Stop sign on left, proceeding with right turn check")
            self.check_right_before_proceeding()
        elif stop_info['x'] > center_x + 50:
            print("Stop sign on right, proceeding with left turn check")
            self.check_left_before_proceeding()
        else:
            print("Stop sign ahead, checking both directions")
            self.check_both_directions_before_proceeding()
            
    def check_right_before_proceeding(self) -> None:
        """Check right side before proceeding from stop."""
        self.px.set_cam_pan_angle(45)
        time.sleep(0.5)
        distance = self.px.ultrasonic.read()
        if distance and distance > self.SAFE_DISTANCE:
            print("Right side clear, proceeding")
            self.px.forward(self.SPEED)
        else:
            print("Obstacle detected, waiting...")
            time.sleep(1)
            
    def check_left_before_proceeding(self) -> None:
        """Check left side before proceeding from stop."""
        self.px.set_cam_pan_angle(-45)
        time.sleep(0.5)
        distance = self.px.ultrasonic.read()
        if distance and distance > self.SAFE_DISTANCE:
            print("Left side clear, proceeding")
            self.px.forward(self.SPEED)
        else:
            print("Obstacle detected, waiting...")
            time.sleep(1)
            
    def check_both_directions_before_proceeding(self) -> None:
        """Check both directions before proceeding from stop."""
        # Check left
        self.px.set_cam_pan_angle(-45)
        time.sleep(0.5)
        left_distance = self.px.ultrasonic.read()
        
        # Check right
        self.px.set_cam_pan_angle(45)
        time.sleep(0.5)
        right_distance = self.px.ultrasonic.read()
        
        # Reset camera
        self.px.set_cam_pan_angle(0)
        
        if (left_distance and right_distance and 
            left_distance > self.SAFE_DISTANCE and 
            right_distance > self.SAFE_DISTANCE):
            print("Both sides clear, proceeding")
            self.px.forward(self.SPEED)
        else:
            print("Obstacles detected, waiting...")
            time.sleep(1)
            
    def set_goal(self, x_offset: float, y_offset: float) -> None:
        """Set goal position as offset from current position in cm."""
        goal_x = self.car_pos[0] + x_offset
        goal_y = self.car_pos[1] + y_offset
        self.goal_pos = np.array([goal_x, goal_y])
        print(f"Goal set at offset ({x_offset}, {y_offset}) cm from current position")
        
    def heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate heuristic (Manhattan distance) between points."""
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
        
    def get_neighbors(self, pos: np.ndarray) -> List[np.ndarray]:
        """Get valid neighboring positions."""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), 
                       (1, 1), (1, -1), (-1, 1), (-1, -1)]:  # 8-directional movement
            new_pos = pos + np.array([dx, dy], dtype=np.float64)  # Ensure float64 type
            x, y = int(new_pos[0]), int(new_pos[1])  # Convert to int for grid indexing
            
            # Check bounds and obstacles
            if (0 <= x < self.grid_size and 0 <= y < self.grid_size and 
                self.grid[y, x] == 0):  # No obstacle
                neighbors.append(new_pos)
        return neighbors
        
    def find_path(self) -> List[np.ndarray]:
        """Find path to goal using A* algorithm."""
        if self.goal_pos is None:
            print("No goal set!")
            return []
            
        # Convert positions to float64 for calculations
        start_pos = self.car_pos.astype(np.float64)
        goal_pos = self.goal_pos.astype(np.float64)
        
        # Convert to tuple for dictionary keys
        start = tuple(start_pos.astype(int))
        goal = tuple(goal_pos.astype(int))
        
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0.0}  # Use float for costs
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            current_pos = np.array(current, dtype=np.float64)
            
            if current == goal:
                break
                
            for next_pos in self.get_neighbors(current_pos):
                next_tuple = tuple(next_pos.astype(int))
                new_cost = cost_so_far[current] + np.linalg.norm(next_pos - current_pos)
                
                if next_tuple not in cost_so_far or new_cost < cost_so_far[next_tuple]:
                    cost_so_far[next_tuple] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goal_pos)
                    heapq.heappush(frontier, (priority, next_tuple))
                    came_from[next_tuple] = current
        
        # Reconstruct path
        path = []
        current = goal
        while current is not None:
            path.append(np.array(current, dtype=np.float64))
            current = came_from.get(current)
        path.reverse()
        
        return path if path else []
        
    def update_position(self, movement: np.ndarray, angle_change: float = 0) -> None:
        """Update car's position and angle based on movement."""
        self.car_pos = self.car_pos.astype(np.float64) + movement.astype(np.float64)
        self.car_angle = (self.car_angle + angle_change) % 360
        
    def get_turn_angle(self, target_pos: np.ndarray) -> float:
        """Calculate turn angle needed to face target position."""
        dx = target_pos[0] - self.car_pos[0]
        dy = target_pos[1] - self.car_pos[1]
        target_angle = math.degrees(math.atan2(dy, dx))
        angle_diff = target_angle - self.car_angle
        
        # Normalize to [-180, 180]
        angle_diff = (angle_diff + 180) % 360 - 180
        return angle_diff
        
    def navigate(self) -> None:
        """Main navigation loop with pathfinding."""
        print("Starting autonomous navigation...")
        
        # Initialize FPS tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        fps_update_interval = 1.0  # Update FPS every second
        last_fps_update = start_time
        last_scan_time = start_time
        SCAN_INTERVAL = 0.5  # Scan environment every 0.5 seconds instead of every frame
        
        # Set goal straight ahead (150cm forward)
        self.set_goal(0, 150)  # 1.5 meters straight ahead
        print("Goal set: 1.5 meters straight ahead")
        
        try:
            while True:
                loop_start = time.time()
                
                # Update frame count and FPS
                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - last_fps_update
                
                if elapsed_time >= fps_update_interval:
                    fps = frame_count / elapsed_time
                    print(f"\rFPS: {fps:.1f}", end='')
                    frame_count = 0
                    last_fps_update = current_time
                
                # Always check front distance first
                front_distance = self.px.ultrasonic.read()
                if front_distance is None:
                    front_distance = 300  # Default to max distance if reading fails
                print(f" | Front distance: {front_distance:.1f}cm", end='')
                
                # Check if goal is reached
                if self.goal_pos is not None:
                    distance_to_goal = np.linalg.norm(self.goal_pos - self.car_pos)
                    print(f" | Distance to goal: {distance_to_goal:.1f}cm", end='')
                    if distance_to_goal < self.GOAL_THRESHOLD:
                        print("\nGoal reached!")
                        self.px.stop()
                        break
                
                # Safety check - stop immediately if too close or invalid reading
                if front_distance < self.DANGER_DISTANCE:
                    print(f"\nDanger! Obstacle detected at {front_distance:.1f}cm!")
                    self.px.stop()
                    self.handle_obstacle()
                    # Replan path after obstacle avoidance
                    self.current_path = self.find_path()
                    self.path_index = 0
                    continue
                
                # Update map and check for path validity only at intervals
                current_time = time.time()
                if current_time - last_scan_time >= SCAN_INTERVAL:
                    readings = self.scan_environment()
                    last_scan_time = current_time
                    path_needs_update = False
                    
                    # Check if current path is blocked
                    if self.current_path:
                        for distance, angle in readings:
                            # Only consider obstacles in front cone
                            if abs(angle) < 45 and distance < self.SAFE_DISTANCE:
                                path_needs_update = True
                                break
                                
                    if path_needs_update or not self.current_path:
                        print("\nReplanning path...")
                        self.current_path = self.find_path()
                        self.path_index = 0
                        if not self.current_path:
                            print("No valid path found!")
                            self.handle_obstacle()
                            continue
                
                # Follow current path
                if self.current_path and self.path_index < len(self.current_path):
                    target = self.current_path[self.path_index]
                    angle_to_target = self.get_turn_angle(target)
                    
                    # Turn towards target
                    if abs(angle_to_target) > 5:  # Reduced angle threshold
                        turn_angle = max(min(angle_to_target, 20), -20)  # Reduced max turn angle
                        self.px.set_dir_servo_angle(turn_angle)
                        self.px.forward(self.TURN_SPEED)
                    else:
                        self.px.set_dir_servo_angle(0)
                        self.px.forward(self.SPEED)
                    
                    # Update position (simplified)
                    movement = np.array([0, 0.5], dtype=np.float64)  # Simplified movement tracking for straight line
                    self.update_position(movement)
                    
                    # Check if reached current waypoint
                    if np.linalg.norm(target - self.car_pos) < self.GOAL_THRESHOLD:
                        self.path_index += 1
                        print("\nReached waypoint", self.path_index)
                else:
                    # No path or reached end of path, stop and replan
                    self.px.stop()
                    self.current_path = self.find_path()
                    self.path_index = 0
                
                # Ensure minimum frame time to prevent CPU overload
                frame_time = time.time() - loop_start
                if frame_time < 0.01:  # Aim for maximum 100 FPS
                    time.sleep(0.01 - frame_time)
                
        except KeyboardInterrupt:
            print("\nNavigation stopped by user")
        except Exception as e:
            print(f"\nProgram error: {e}")
            raise  # Re-raise the exception for debugging
        finally:
            self.cleanup()
            
    def handle_obstacle(self) -> None:
        """Handle detected obstacle."""
        print("\nHandling obstacle...")
        self.px.stop()  # Make sure we're stopped
        
        # Check left side
        self.px.set_cam_pan_angle(-45)
        time.sleep(0.3)  # Reduced delay
        left_distances = []
        for _ in range(3):  # Take multiple readings
            dist = self.px.ultrasonic.read()
            if dist is not None and dist > 0:
                left_distances.append(dist)
            time.sleep(0.01)
        left_distance = min(left_distances) if left_distances else 0
        
        # Check right side
        self.px.set_cam_pan_angle(45)
        time.sleep(0.3)
        right_distances = []
        for _ in range(3):
            dist = self.px.ultrasonic.read()
            if dist is not None and dist > 0:
                right_distances.append(dist)
            time.sleep(0.01)
        right_distance = min(right_distances) if right_distances else 0
        
        # Reset camera
        self.px.set_cam_pan_angle(0)
        
        print(f"Left distance: {left_distance}cm, Right distance: {right_distance}cm")
        
        # Choose direction with more space
        if left_distance > right_distance and left_distance > self.SAFE_DISTANCE:
            print("Turning left to avoid obstacle")
            self.px.set_dir_servo_angle(-30)
            self.px.forward(self.TURN_SPEED)
            time.sleep(0.5)
        elif right_distance > self.SAFE_DISTANCE:
            print("Turning right to avoid obstacle")
            self.px.set_dir_servo_angle(30)
            self.px.forward(self.TURN_SPEED)
            time.sleep(0.5)
        else:
            print("No clear path, backing up")
            self.px.backward(self.SPEED)
            time.sleep(1)
            # Try turning after backing up
            if left_distance > right_distance:
                self.px.set_dir_servo_angle(-45)
            else:
                self.px.set_dir_servo_angle(45)
            time.sleep(0.5)
        
        # Reset steering
        self.px.set_dir_servo_angle(0)
        self.px.stop()
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self.px.stop()
        self.px.set_cam_pan_angle(0)
        Vilib.object_detect_switch(False)
        Vilib.camera_close()

def main():
    print("Initializing autonomous navigation system...")
    navigator = AutonomousNavigator()
    
    try:
        navigator.navigate()
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program error: {e}")
    finally:
        print("Cleaning up...")
        navigator.cleanup()
        print("Shutdown complete")

if __name__ == '__main__':
    main() 