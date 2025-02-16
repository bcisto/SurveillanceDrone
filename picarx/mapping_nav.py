import time
import math
import numpy as np
from picarx import Picarx
from queue import PriorityQueue
from typing import List, Tuple, Set
import tflite_runtime.interpreter as tflite
import cv2
import os

class Navigator:
    def __init__(self):
        self.px = Picarx()
        self.current_x = 0  # inches
        self.current_y = 0  # inches
        self.current_angle = 0  # degrees, 0 is north
        
        # Constants for movement
        self.SPEED = 25  # Reduced speed for better control
        self.TURN_SPEED = 20
        self.SCAN_STEP = 5
        
        # Distance thresholds (in cm)
        self.SAFE_DISTANCE = 40    # Safe distance to maintain
        self.DANGER_DISTANCE = 20  # Distance to trigger reverse
        self.TURN_DISTANCE = 30    # Distance to start turning
        
        # Physical dimensions (in cm)
        self.WHEEL_DIAMETER = 6.5
        self.WHEEL_BASE = 11.5
        self.CAR_WIDTH = 16.5      # Full width of car
        self.CAR_LENGTH = 25.4     # Full length of car
        
        # Camera panning
        self.camera_pan_angle = 0
        self.PAN_SPEED = 1
        self.PAN_RANGE = 60
        self.panning_right = True
        self.last_pan_time = time.time()
        self.last_scan_angle = 0
        self.stop_panning = False  # Flag to control camera panning
        
        # Path planning settings
        self.REPLAN_STEPS = 2      # Reduced for more frequent replanning
        self.CLEARANCE_RADIUS = int(self.CAR_WIDTH/2.54) + 2  # Convert to inches and add safety margin
        
        # Grid for mapping (1 cell = 1 inch)
        self.grid_size = 200
        self.grid = np.zeros((self.grid_size, self.grid_size))
        
        # Initialize other components
        self.interpreter = self.load_person_detector()
        if self.interpreter:
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_shape = self.input_details[0]['shape']
        
        self.movements = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.camera = cv2.VideoCapture(0)
        
        # Give time for robot to initialize
        time.sleep(0.5)
        
        # Set initial servo position
        self.px.set_cam_pan_angle(0)
        
    def load_person_detector(self):
        """Load and return TensorFlow Lite model for person detection."""
        try:
            model_path = os.path.join(os.path.dirname(__file__), 'models', 'efficientdet_lite0.tflite')
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            print("Successfully loaded person detection model")
            return interpreter
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
        
    def preprocess_image(self, image):
        """Preprocess image for the EfficientDet model."""
        input_size = self.input_shape[1:3]
        resized = cv2.resize(image, input_size)
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)
        
    def check_for_person(self):
        """Check if a person is detected in front of the car using TF Lite model."""
        if not self.interpreter:
            return False
            
        ret, frame = self.camera.read()
        if not ret:
            return False
            
        # Preprocess image
        processed_image = self.preprocess_image(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
        self.interpreter.invoke()
        
        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        
        # Check for person detection (class 0 is typically person in COCO dataset)
        for i in range(len(scores[0])):
            if scores[0][i] > 0.5 and classes[0][i] == 0:  # Confidence threshold of 0.5
                return True
                
        return False
        
    def scan_surroundings(self):
        """Scan surroundings with improved obstacle detection and width consideration."""
        print("Scanning surroundings...")
        
        # Reset grid in visible area
        visible_radius = 50  # inches
        center_x = int(self.current_x + self.grid_size/2)
        center_y = int(self.current_y + self.grid_size/2)
        for y in range(max(0, center_y-visible_radius), min(self.grid_size, center_y+visible_radius)):
            for x in range(max(0, center_x-visible_radius), min(self.grid_size, center_x+visible_radius)):
                if math.sqrt((x-center_x)**2 + (y-center_y)**2) <= visible_radius:
                    self.grid[y, x] = 0
        
        # Take multiple readings at current camera angle
        readings = []
        for _ in range(3):  # Take 3 readings
            distance = self.px.ultrasonic.read()
            if distance > 0 and distance < 300:
                readings.append(distance)
            time.sleep(0.02)
            
        if readings:
            # Use minimum distance for safety
            distance = min(readings)
            
            # Convert to relative x,y coordinates using current camera angle
            rad_angle = math.radians(self.camera_pan_angle + self.current_angle)
            rel_x = distance * math.sin(rad_angle)
            rel_y = distance * math.cos(rad_angle)
            
            # Convert to absolute coordinates and grid coordinates
            abs_x = self.current_x + rel_x
            abs_y = self.current_y + rel_y
            grid_x = int(abs_x + self.grid_size/2)
            grid_y = int(abs_y + self.grid_size/2)
            
            # Mark obstacle with improved clearance
            self._mark_obstacle_with_clearance(grid_x, grid_y)
        
        # Store last scan angle
        self.last_scan_angle = self.camera_pan_angle
        
    def _mark_obstacle_with_clearance(self, grid_x: int, grid_y: int):
        """Mark an obstacle and its clearance area with improved width consideration."""
        # Calculate clearance based on car width plus safety margin
        car_width_inches = self.CAR_WIDTH / 2.54  # Convert cm to inches
        clearance_radius = int(car_width_inches + 4)  # Add 4 inches safety margin
        
        # Mark core obstacle
        core_radius = int(car_width_inches / 2)  # Use half car width as core
        for dy in range(-core_radius, core_radius + 1):
            for dx in range(-core_radius, core_radius + 1):
                if dx*dx + dy*dy <= core_radius*core_radius:
                    x, y = grid_x + dx, grid_y + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.grid[y, x] = 1
        
        # Mark graduated clearance area
        for dy in range(-clearance_radius, clearance_radius + 1):
            for dx in range(-clearance_radius, clearance_radius + 1):
                dist_sq = dx*dx + dy*dy
                if dist_sq <= clearance_radius*clearance_radius:
                    x, y = grid_x + dx, grid_y + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        # Create graduated clearance values
                        if dist_sq <= core_radius*core_radius:
                            self.grid[y, x] = 1.0  # Core obstacle
                        else:
                            # Calculate graduated value based on distance
                            dist_ratio = math.sqrt(dist_sq) / clearance_radius
                            clearance_value = max(0.4, 1.0 - dist_ratio)
                            self.grid[y, x] = max(self.grid[y, x], clearance_value)
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def simulate_car_movement(self, x: float, y: float, heading: float, steering_angle: float, distance: float, reverse: bool = False) -> Tuple[float, float, float]:
        """Simulate car movement using bicycle model."""
        # Convert angles to radians
        heading_rad = math.radians(heading)
        steering_rad = math.radians(steering_angle)
        
        # If reversing, negate the distance
        if reverse:
            distance = -distance
            
        # Calculate new position and heading using bicycle model
        if abs(steering_angle) < 0.001:  # Driving straight
            new_x = x + distance * math.sin(heading_rad)
            new_y = y + distance * math.cos(heading_rad)
            new_heading = heading
        else:
            # Calculate turning radius
            turning_radius = self.WHEEL_BASE / math.tan(steering_rad)
            # Calculate change in heading
            d_heading = distance / turning_radius
            if reverse:
                d_heading = -d_heading
            new_heading = (heading + math.degrees(d_heading)) % 360
            
            # Calculate new position
            new_x = x + turning_radius * (math.cos(heading_rad) - math.cos(heading_rad + d_heading))
            new_y = y + turning_radius * (math.sin(heading_rad + d_heading) - math.sin(heading_rad))
            
        return new_x, new_y, new_heading

    def get_steering_angles(self) -> List[float]:
        """Get possible steering angles for path planning."""
        return [-40, -20, 0, 20, 40]  # Degrees

    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find path using Hybrid A* algorithm with improved obstacle avoidance and timeout protection."""
        # Add timeout protection
        start_time = time.time()
        MAX_SEARCH_TIME = 5.0  # Maximum 5 seconds for path finding
        
        # Node structure: (x, y, heading, steering_angle, is_reverse, parent)
        start_node = (float(start[0]), float(start[1]), self.current_angle, 0, False, None)
        goal_pos = (float(goal[0]), float(goal[1]))
        
        # Priority queue for open nodes: (f_cost, node)
        frontier = PriorityQueue()
        frontier.put((0, start_node))
        
        # Dictionary to store cost to reach each node
        cost_so_far = {(start_node[0], start_node[1], start_node[2]): 0}
        
        # Dictionary to track visited cells with headings
        visited = {}  # (cell_x, cell_y, rounded_heading) -> True
        
        # Movement distance per step (increased for faster exploration)
        step_distance = 5.0  # inches (increased from 3.0)
        
        # Heading resolution for visited cells (increased for faster exploration)
        heading_resolution = 30  # degrees (increased from 15)
        
        # Reduced number of steering angles for faster exploration
        steering_angles = [-30, 0, 30]  # Reduced from [-40, -20, 0, 20, 40]
        
        while not frontier.empty():
            # Check timeout
            if time.time() - start_time > MAX_SEARCH_TIME:
                print("Path finding timed out, returning best partial path...")
                # Return path to the best node found so far
                best_node = min([(math.sqrt((n[0] - goal_pos[0])**2 + (n[1] - goal_pos[1])**2), n) 
                               for n in cost_so_far.keys()], key=lambda x: x[0])[1]
                path = []
                node = next(n[1] for n in frontier.queue if (n[1][0], n[1][1], n[1][2]) == best_node)
                while node is not None:
                    path.append((int(node[0]), int(node[1])))
                    node = node[5]  # parent
                path.reverse()
                return path
            
            current = frontier.get()[1]
            current_x, current_y, current_heading, current_steering, is_reverse, _ = current
            
            # Check if we've reached the goal
            dist_to_goal = math.sqrt((current_x - goal_pos[0])**2 + (current_y - goal_pos[1])**2)
            if dist_to_goal < 15:  # Increased from 10 to 15 inches for easier goal reaching
                path = []
                node = current
                while node is not None:
                    path.append((int(node[0]), int(node[1])))
                    node = node[5]  # parent
                path.reverse()
                return path
            
            # Try different steering angles and directions
            for steering_angle in steering_angles:  # Using reduced steering angles
                for reverse in [False, True]:
                    # Skip reverse if we're close to the goal
                    if reverse and dist_to_goal < 30:
                        continue
                        
                    # Simulate car movement
                    new_x, new_y, new_heading = self.simulate_car_movement(
                        current_x, current_y, current_heading, 
                        steering_angle, step_distance, reverse
                    )
                    
                    # Convert to grid cell
                    cell_x, cell_y = int(new_x), int(new_y)
                    
                    # Quick bounds check before detailed collision check
                    if not (0 <= cell_x < self.grid_size and 0 <= cell_y < self.grid_size):
                        continue
                    
                    # Quick obstacle check before detailed collision check
                    if self.grid[cell_y, cell_x] > 0.8:  # If directly on high-value obstacle
                        continue
                    
                    # Check for collision along the car's body
                    if self._check_collision(new_x, new_y, new_heading):
                        continue
                    
                    # Round heading for visited check
                    rounded_heading = round(new_heading / heading_resolution) * heading_resolution
                    cell_key = (cell_x, cell_y, rounded_heading)
                    
                    if cell_key in visited:
                        continue
                    
                    # Calculate costs with increased penalties for risky moves
                    new_cost = cost_so_far[(current_x, current_y, current_heading)] + step_distance
                    
                    # Add penalties
                    if reverse:
                        new_cost += 5  # Reduced from 8
                    if steering_angle != current_steering:
                        new_cost += 2  # Reduced from 3
                    
                    # Add penalty for proximity to obstacles
                    proximity_cost = self._calculate_proximity_cost(new_x, new_y)
                    new_cost += proximity_cost * 0.5  # Reduced weight of proximity cost
                    
                    # Calculate heuristic with reduced weight
                    h_cost = math.sqrt((cell_x - goal_pos[0])**2 + (cell_y - goal_pos[1])**2) * 0.8
                    f_cost = new_cost + h_cost
                    
                    # Create new node
                    new_node = (new_x, new_y, new_heading, steering_angle, reverse, current)
                    node_key = (new_x, new_y, new_heading)
                    
                    if node_key not in cost_so_far or new_cost < cost_so_far[node_key]:
                        cost_so_far[node_key] = new_cost
                        frontier.put((f_cost, new_node))
                        visited[cell_key] = True
        
        return []  # No path found

    def _check_collision(self, x: float, y: float, heading: float) -> bool:
        """Check if the car collides with any obstacles at the given position and heading."""
        # Convert to grid coordinates
        grid_x, grid_y = int(x), int(y)
        
        # Calculate car corners based on heading
        rad_heading = math.radians(heading)
        cos_h = math.cos(rad_heading)
        sin_h = math.sin(rad_heading)
        
        # Check points along the car's body
        for dx in np.linspace(-self.CAR_LENGTH/2, self.CAR_LENGTH/2, 5):
            for dy in np.linspace(-self.CAR_WIDTH/2, self.CAR_WIDTH/2, 3):
                # Rotate point by heading
                rotated_x = dx * cos_h - dy * sin_h
                rotated_y = dx * sin_h + dy * cos_h
                
                # Add to car's position and convert to grid
                check_x = int(grid_x + rotated_x)
                check_y = int(grid_y + rotated_y)
                
                # Check if point is in obstacle
                if (0 <= check_x < self.grid_size and 
                    0 <= check_y < self.grid_size and 
                    self.grid[check_y, check_x] > 0.5):
                    return True
        
        return False

    def _calculate_proximity_cost(self, x: float, y: float) -> float:
        """Calculate cost based on proximity to obstacles."""
        grid_x, grid_y = int(x), int(y)
        proximity_cost = 0
        
        # Check surrounding cells
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                check_x = grid_x + dx
                check_y = grid_y + dy
                if (0 <= check_x < self.grid_size and 
                    0 <= check_y < self.grid_size):
                    # Add cost based on obstacle value and distance
                    obstacle_value = self.grid[check_y, check_x]
                    if obstacle_value > 0:
                        distance = math.sqrt(dx*dx + dy*dy)
                        proximity_cost += (obstacle_value * (3 - distance)) if distance < 3 else 0
        
        return proximity_cost

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (inches) to grid coordinates, clamping to valid grid indices."""
        # Convert to grid coordinates
        grid_x = int(x + self.grid_size/2)
        grid_y = int(y + self.grid_size/2)
        
        # Clamp to valid grid indices
        grid_x = max(0, min(grid_x, self.grid_size - 1))
        grid_y = max(0, min(grid_y, self.grid_size - 1))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (inches)."""
        world_x = grid_x - self.grid_size/2
        world_y = grid_y - self.grid_size/2
        return world_x, world_y
    
    def print_grid(self):
        """Print the current grid state for debugging."""
        print("\nCurrent Grid Map:")
        print("■ = obstacle, · = clear, S = start, G = goal")
        
        # Get current position and goal in grid coordinates
        start_x, start_y = self.world_to_grid(self.current_x, self.current_y)
        
        # Print grid with current position
        for y in range(max(0, start_y-20), min(self.grid_size, start_y+20)):
            row = ""
            for x in range(max(0, start_x-20), min(self.grid_size, start_x+20)):
                if x == start_x and y == start_y:
                    row += "S"
                elif self.grid[y, x] == 1:
                    row += "■"
                else:
                    row += "·"
            print(row)
        
        print(f"\nGrid size: {self.grid_size}x{self.grid_size} cells")
        print(f"Current position (world): ({self.current_x:.1f}, {self.current_y:.1f}) inches")
        print(f"Current position (grid): ({start_x}, {start_y})")
    
    def update_camera_pan(self):
        """Update camera pan angle for continuous scanning."""
        if self.stop_panning:  # Don't update if panning should be stopped
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.last_pan_time
        
        # Update pan angle based on elapsed time for smooth motion
        if elapsed_time >= 0.02:  # 50Hz update rate
            # Calculate new angle based on time elapsed
            angle_change = self.PAN_SPEED * elapsed_time * 50  # Scale by update rate
            
            if self.panning_right:
                self.camera_pan_angle = min(self.camera_pan_angle + angle_change, self.PAN_RANGE)
                if self.camera_pan_angle >= self.PAN_RANGE:
                    self.panning_right = False
            else:
                self.camera_pan_angle = max(self.camera_pan_angle - angle_change, -self.PAN_RANGE)
                if self.camera_pan_angle <= -self.PAN_RANGE:
                    self.panning_right = True
            
            # Update servo position
            self.px.set_cam_pan_angle(int(self.camera_pan_angle))
            self.last_pan_time = current_time

    def handle_blocked_path(self) -> bool:
        """Handle situation when path is blocked with improved distance-based behavior and width analysis."""
        print("Path blocked, checking surroundings...")
        
        # First, back up 40 cm to give more room to maneuver
        print("Backing up 40 cm to get more room...")
        self.move_distance(-4)  # 3 inches/10.16 cm
        
        # Collect readings from all directions while panning
        readings = []
        start_time = time.time()
        while time.time() - start_time < 1.0:  # Collect readings for 1 second
            self.update_camera_pan()
            distance = self.px.ultrasonic.read()
            if 0 < distance < 300:  # Valid reading range
                readings.append((self.camera_pan_angle, distance))
            time.sleep(0.02)
        
        if not readings:
            print("No valid readings obtained")
            return False
            
        # Sort readings by angle for sequential analysis
        readings.sort(key=lambda x: x[0])
        
        # Process readings by direction with width analysis
        front_readings = []  # (angle, distance) pairs
        left_readings = []   # (angle, distance) pairs
        right_readings = []  # (angle, distance) pairs
        
        for angle, distance in readings:
            if -15 <= angle <= 15:  # Front sector
                front_readings.append((angle, distance))
            elif angle < -15:  # Left sector
                left_readings.append((angle, distance))
            else:  # Right sector
                right_readings.append((angle, distance))
        
        # Analyze obstacle widths and closest points
        def analyze_sector(sector_readings):
            if not sector_readings:
                return float('inf'), 0, None
            
            min_dist = float('inf')
            max_width = 0
            last_angle = None
            last_dist = None
            width_at_min_dist = 0
            
            for angle, dist in sector_readings:
                if dist < min_dist:
                    min_dist = dist
                    width_at_min_dist = 0
                
                if last_angle is not None and last_dist is not None:
                    # If consecutive readings are close in distance, they might be part of the same obstacle
                    if abs(dist - last_dist) < 20:  # 20cm threshold for same obstacle
                        # Calculate approximate width using trigonometry
                        angle_diff = abs(angle - last_angle)
                        avg_dist = (dist + last_dist) / 2
                        width = 2 * avg_dist * math.sin(math.radians(angle_diff/2))
                        max_width = max(max_width, width)
                        
                        if abs(dist - min_dist) < 10:  # Within 10cm of minimum distance
                            width_at_min_dist = max(width_at_min_dist, width)
                
                last_angle = angle
                last_dist = dist
            
            return min_dist, max_width, width_at_min_dist
        
        # Analyze each sector
        front_dist, front_width, front_width_at_min = analyze_sector(front_readings)
        left_dist, left_width, left_width_at_min = analyze_sector(left_readings)
        right_dist, right_width, right_width_at_min = analyze_sector(right_readings)
        
        print(f"Front - Distance: {front_dist:.1f}cm, Width: {front_width:.1f}cm")
        print(f"Left - Distance: {left_dist:.1f}cm, Width: {left_width:.1f}cm")
        print(f"Right - Distance: {right_dist:.1f}cm, Width: {right_width:.1f}cm")
        
        # Calculate required clearance based on car width plus safety margin
        required_clearance = self.CAR_WIDTH + 10  # Add 10cm safety margin
        
        # Decision making based on distances and widths
        if front_dist < self.TURN_DISTANCE:
            # Calculate turn angle based on obstacle width
            turn_angle = min(60, max(30, int(front_width / required_clearance * 45)))
            
            # Choose turn direction based on available space and obstacle widths
            if (left_dist > right_dist and left_dist > self.SAFE_DISTANCE and 
                left_width < required_clearance):
                print(f"Obstacle ahead, turning left (angle: {turn_angle}°)")
                self.turn_angle(-turn_angle)
                return True
            elif (right_dist > self.SAFE_DISTANCE and right_width < required_clearance):
                print(f"Obstacle ahead, turning right (angle: {turn_angle}°)")
                self.turn_angle(turn_angle)
                return True
            else:
                print("No clear path available - spaces too narrow")
                self.stop_panning = True
                time.sleep(0.1)
                return False
                
        elif front_dist < self.SAFE_DISTANCE:
            # Make gentler turns when obstacle is further but still consider width
            turn_angle = min(45, max(20, int(front_width / required_clearance * 30)))
            
            if (left_dist > right_dist and left_dist > self.SAFE_DISTANCE and 
                left_width < required_clearance):
                print(f"Preemptive turn left (angle: {turn_angle}°)")
                self.turn_angle(-turn_angle)
                return True
            elif (right_dist > self.SAFE_DISTANCE and right_width < required_clearance):
                print(f"Preemptive turn right (angle: {turn_angle}°)")
                self.turn_angle(turn_angle)
                return True
        
        return True  # Return True to allow replanning after backing up

    def cleanup_servos(self):
        """Reset all servos to their zero positions."""
        print("Zeroing all servos...")
        
        # Stop camera panning immediately
        self.stop_panning = True
        time.sleep(0.1)  # Give time for panning to stop
        
        # Force stop any ongoing servo movement
        self.px.stop()
        
        # Reset camera pan with a smooth motion
        current_angle = self.camera_pan_angle
        steps = 10
        for i in range(steps):
            target_angle = int(current_angle * (steps - i - 1) / steps)
            self.px.set_cam_pan_angle(target_angle)
            time.sleep(0.05)  # Increased delay for more reliable servo movement
        
        # Ensure camera is at zero position with multiple attempts
        for _ in range(3):  # Try up to 3 times
            self.camera_pan_angle = 0
            self.px.set_cam_pan_angle(0)
            time.sleep(0.1)  # Wait between attempts
        
        # Reset steering
        self.px.set_dir_servo_angle(0)
        time.sleep(0.1)
        
        # Final stop of all movement
        self.px.stop()
        
        # Give servos time to reach final position
        time.sleep(0.5)
        
        # One final force to zero position
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)

    def navigate_to_goal(self, goal_x: float, goal_y: float):
        """Navigate to goal using A* pathfinding with periodic replanning."""
        print(f"Navigating to goal: ({goal_x}, {goal_y}) inches")
        
        # Reset panning flag at start of navigation
        self.stop_panning = False
        
        # Check if goal is within bounds before starting
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_x, goal_y)
        if not (0 <= goal_grid_x < self.grid_size and 0 <= goal_grid_y < self.grid_size):
            print(f"Error: Goal position ({goal_x}, {goal_y}) is outside navigable area!")
            print(f"Please choose a goal within ±{self.grid_size//2} inches of start position.")
            return
        
        consecutive_failures = 0
        navigation_complete = False
        
        try:
            while not navigation_complete:
                # Update camera pan continuously
                self.update_camera_pan()
                
                # Convert current position and goal to grid coordinates
                start_grid_x, start_grid_y = self.world_to_grid(self.current_x, self.current_y)
                goal_grid_x, goal_grid_y = self.world_to_grid(goal_x, goal_y)
                
                print(f"\nCurrent position (world): ({self.current_x:.1f}, {self.current_y:.1f}) inches")
                print(f"Current position (grid): ({start_grid_x}, {start_grid_y})")
                print(f"Goal position (world): ({goal_x:.1f}, {goal_y:.1f}) inches")
                print(f"Goal position (grid): ({goal_grid_x}, {goal_grid_y})")
                
                # Check if we've reached the goal
                distance_to_goal = math.sqrt((goal_x - self.current_x)**2 + (goal_y - self.current_y)**2)
                if distance_to_goal < 5:  # Within 5 inches of goal
                    print("Reached goal!")
                    navigation_complete = True
                    self.stop_panning = True  # Stop panning before cleanup
                    time.sleep(0.1)  # Give time for panning to stop
                    self.cleanup_servos()  # Zero all servos when reaching goal
                    return  # Exit the method completely
                
                # Scan surroundings and update map
                print("\nScanning surroundings...")
                self.scan_surroundings()
                
                # Print current grid state
                self.print_grid()
                
                # Check if start or goal is in obstacle
                if self.grid[start_grid_y, start_grid_x] == 1:
                    if not self.handle_blocked_path():
                        print("Error: Start position is blocked and cannot find alternative!")
                        self.stop_panning = True  # Stop panning before breaking
                        time.sleep(0.1)  # Give time for panning to stop
                        break
                    consecutive_failures = 0
                    continue
                    
                if self.grid[goal_grid_y, goal_grid_x] == 1:
                    print("Error: Goal position is in obstacle!")
                    self.stop_panning = True  # Stop panning before breaking
                    time.sleep(0.1)  # Give time for panning to stop
                    break
                
                # Find path to goal
                print("\nFinding path...")
                path = self.find_path((start_grid_x, start_grid_y), (goal_grid_x, goal_grid_y))
                
                if not path:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        if not self.handle_blocked_path():
                            print("No path found after multiple attempts. Stopping navigation.")
                            self.stop_panning = True  # Stop panning before breaking
                            time.sleep(0.1)  # Give time for panning to stop
                            break
                        consecutive_failures = 0
                    continue
                
                consecutive_failures = 0
                print(f"Found path with {len(path)} steps")
                
                # Follow path for REPLAN_STEPS steps or until blocked
                for i, (next_x, next_y) in enumerate(path[1:self.REPLAN_STEPS+1]):
                    # Update camera pan while moving
                    self.update_camera_pan()
                    
                    print(f"\nMoving to point {i+1}: ({next_x}, {next_y})")
                    
                    # Check for person
                    if self.check_for_person():
                        print("Person detected! Waiting...")
                        while self.check_for_person():
                            self.update_camera_pan()
                            time.sleep(0.5)
                        print("Path clear, continuing...")
                        break
                    
                    # Calculate movement
                    dx = next_x - (self.current_x + self.grid_size/2)
                    dy = next_y - (self.current_y + self.grid_size/2)
                    
                    # Calculate angle to next point
                    angle_to_next = math.degrees(math.atan2(dx, dy))
                    angle_diff = angle_to_next - self.current_angle
                    angle_diff = (angle_diff + 180) % 360 - 180
                    
                    # Turn towards next point
                    if abs(angle_diff) > 5:
                        print(f"Turning {angle_diff:.1f} degrees")
                        self.turn_angle(angle_diff)
                    
                    # Move to next point
                    distance = math.sqrt(dx*dx + dy*dy)
                    print(f"Moving {distance:.1f} inches")
                    if not self.move_distance(distance):
                        print("Movement interrupted by obstacle - replanning...")
                        # Mark the area as blocked with bounds checking
                        grid_x, grid_y = self.world_to_grid(next_x, next_y)
                        if (0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size):
                            self.grid[grid_y, grid_x] = 1
                            # Also mark surrounding cells as blocked for safety
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    check_x = grid_x + dx
                                    check_y = grid_y + dy
                                    if (0 <= check_x < self.grid_size and 
                                        0 <= check_y < self.grid_size):
                                        self.grid[check_y, check_x] = 1
                        else:
                            print("Warning: Obstacle detected outside grid bounds")
                        
                        # Handle blocked path
                        if not self.handle_blocked_path():
                            print("Cannot find alternative path!")
                            self.stop_panning = True
                            time.sleep(0.1)
                            break
                        break  # Break out of path following to force replanning
                    
                    # Break if we've moved enough steps
                    if i >= self.REPLAN_STEPS - 1:
                        break
                
                # Short pause between iterations
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nNavigation interrupted by user")
        finally:
            # Always cleanup servos, even if interrupted
            self.cleanup_servos()
            if self.camera:
                self.camera.release()

    def move_distance(self, distance_inches: float):
        """Move forward/backward a specific distance in inches with continuous obstacle detection."""
        # Convert inches to cm
        distance_cm = distance_inches * 2.54
        
        # Calculate time needed at current speed
        # At speed 30, approximately 20cm/second
        time_needed = abs(distance_cm) / 20.0
        
        # Move
        if distance_inches > 0:
            self.px.forward(self.SPEED)
        else:
            self.px.backward(self.SPEED)
        
        # Keep camera panning while moving and check for obstacles
        start_time = time.time()
        while time.time() - start_time < time_needed:
            self.update_camera_pan()
            
            # Check for obstacles if moving forward
            if distance_inches > 0:
                # First check front
                front_distance = self.px.ultrasonic.read()
                if 0 < front_distance < self.DANGER_DISTANCE:  # If obstacle detected too close
                    print(f"Obstacle detected ahead at {front_distance}cm, checking sides...")
                    self.px.stop()
                    
                    # Check left side
                    self.px.set_cam_pan_angle(-45)  # Look left
                    time.sleep(0.2)  # Give time for servo to move
                    left_distance = self.px.ultrasonic.read()
                    
                    # Check right side
                    self.px.set_cam_pan_angle(45)  # Look right
                    time.sleep(0.2)  # Give time for servo to move
                    right_distance = self.px.ultrasonic.read()
                    
                    # Reset camera
                    self.px.set_cam_pan_angle(0)
                    
                    print(f"Left distance: {left_distance}cm, Right distance: {right_distance}cm")
                    
                    # Decide which way to turn based on side distances
                    if left_distance > self.SAFE_DISTANCE and left_distance > right_distance:
                        print("Clear path on left, turning left")
                        self.turn_angle(-45)
                        return False
                    elif right_distance > self.SAFE_DISTANCE:
                        print("Clear path on right, turning right")
                        self.turn_angle(45)
                        return False
                    else:
                        print("No clear path on sides, backing up")
                        # Back up a bit for safety
                        self.px.backward(self.SPEED)
                        time.sleep(0.5)
                        self.px.stop()
                        
                        # Update position based on partial movement
                        partial_time = time.time() - start_time
                        partial_distance = (partial_time / time_needed) * distance_inches
                        rad_angle = math.radians(self.current_angle)
                        self.current_x += partial_distance * math.sin(rad_angle)
                        self.current_y += partial_distance * math.cos(rad_angle)
                        return False  # Indicate movement was interrupted
                
            time.sleep(0.02)
            
        self.px.stop()
        
        # Update position
        rad_angle = math.radians(self.current_angle)
        self.current_x += distance_inches * math.sin(rad_angle)
        self.current_y += distance_inches * math.cos(rad_angle)
        return True  # Indicate movement completed successfully

    def turn_angle(self, angle_degrees: float):
        """Turn relative to current angle."""
        # Constrain angle to valid range
        angle_degrees = max(min(angle_degrees, self.px.DIR_MAX), self.px.DIR_MIN)
        
        # Set steering angle
        self.px.set_dir_servo_angle(angle_degrees)
        
        # Move forward briefly to execute the turn while keeping camera panning
        self.px.forward(self.SPEED)
        start_time = time.time()
        while time.time() - start_time < 0.5:  # 0.5 second turn
            self.update_camera_pan()
            time.sleep(0.02)
        self.px.stop()
        
        # Update angle
        self.current_angle = (self.current_angle + angle_degrees) % 360
        
        # Reset steering to straight
        self.px.set_dir_servo_angle(0)

def main():
    nav = Navigator()
    
    try:
        # Set goal 5 feet (60 inches) straight ahead
        # Since 0 degrees is north in our coordinate system, and we start at (0,0),
        # we want to go 60 inches in the y direction
        goal_x = 0
        goal_y = 60  # 5 feet = 60 inches
        
        print("\nInitial grid state:")
        nav.print_grid()
        print("\nStarting navigation to goal 5 feet straight ahead...")
        print(f"Goal coordinates: ({goal_x}, {goal_y}) inches")
        
        # Do an initial scan before moving
        print("\nPerforming initial scan...")
        nav.scan_surroundings()
        
        # Start navigation
        nav.navigate_to_goal(goal_x, goal_y)
        
    except KeyboardInterrupt:
        print("\nNavigation interrupted by user")
    finally:
        # Always cleanup servos, even if interrupted
        nav.cleanup_servos()
        if nav.camera:
            nav.camera.release()

if __name__ == '__main__':
    main() 