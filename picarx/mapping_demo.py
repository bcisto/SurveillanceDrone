from picarx import Picarx
import time
import numpy as np
import os
import math

class ObstacleMap:
    def __init__(self, size: int = 100, resolution: float = 1.0, angle_range: tuple = (-90, 90)):
        """
        Initialize both a polar and 2D grid map for obstacle detection.
        
        Args:
            size: Size of the map in cm
            resolution: Resolution in cm per grid cell
            angle_range: Tuple of (min_angle, max_angle) in degrees
        """
        self.size = size
        self.resolution = resolution
        self.grid_size = int(size / resolution)
        
        # Initialize 2D grid map
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.car_pos = np.array([self.grid_size//2, self.grid_size//2])  # Car at center
        
        # Initialize polar map
        self.min_angle, self.max_angle = angle_range
        self.angle_range = self.max_angle - self.min_angle + 1
        self.polar_map = np.zeros(self.angle_range, dtype=np.float32)
        self.polar_map.fill(np.inf)
        
    def update_obstacle(self, distance: float, angle_deg: float) -> None:
        """
        Update both maps with a new obstacle reading.
        
        Args:
            distance: Distance to obstacle in cm
            angle_deg: Angle of the reading in degrees
        """
        if distance <= 0 or distance > self.size:
            return
            
        # Update polar map
        idx = int(angle_deg - self.min_angle)
        if 0 <= idx < self.angle_range:
            self.polar_map[idx] = min(distance, self.polar_map[idx])
        
        # Update 2D grid map
        angle_rad = np.radians(angle_deg)
        grid_distance = distance / self.resolution
        
        # Calculate obstacle position in grid coordinates
        dx = grid_distance * np.cos(angle_rad)
        dy = grid_distance * np.sin(angle_rad)
        
        # Convert to grid coordinates (car is at center)
        obstacle_pos = self.car_pos + np.array([dx, dy])
        x, y = int(obstacle_pos[0]), int(obstacle_pos[1])
        
        # Check bounds and mark obstacle
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1
            
            # Draw line from car to obstacle (Bresenham's line algorithm)
            x0, y0 = int(self.car_pos[0]), int(self.car_pos[1])
            self._draw_line(x0, y0, x, y)
            
    def _draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """
        Draw a line in the grid using Bresenham's algorithm.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.grid[y, x] = 1
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.grid[y, x] = 1
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            self.grid[y, x] = 1
            
    def interpolate_readings(self, readings: list) -> None:
        """
        Interpolate between sensor readings to fill gaps.
        
        Args:
            readings: List of (distance, angle_deg) tuples
        """
        if len(readings) < 2:
            return
            
        # Sort readings by angle
        readings = sorted(readings, key=lambda x: x[1])
        
        # Interpolate between consecutive readings
        for i in range(len(readings)-1):
            dist1, angle1 = readings[i]
            dist2, angle2 = readings[i+1]
            
            # Skip if angles are too far apart
            if abs(angle2 - angle1) > 10:
                continue
                
            # Linear interpolation for both maps
            steps = int(abs(angle2 - angle1) * 2)
            angles = np.linspace(angle1, angle2, steps)
            distances = np.linspace(dist1, dist2, steps)
            
            for angle, distance in zip(angles, distances):
                self.update_obstacle(distance, angle)

def scan_environment(px: Picarx, obstacle_map: ObstacleMap, step: int = 2):
    """
    Perform a continuous scan of the environment using the ultrasonic sensor.
    
    Args:
        px: Picarx instance
        obstacle_map: ObstacleMap instance to update
        step: Angle step size in degrees
    """
    readings = []
    scan_speed = 0.01
    
    # Move to starting position
    for angle in range(0, obstacle_map.min_angle, -1):
        px.set_cam_pan_angle(angle)
        time.sleep(scan_speed)
    
    try:
        while True:
            # Scan from min to max angle
            for angle in range(obstacle_map.min_angle, obstacle_map.max_angle + 1, step):
                px.set_cam_pan_angle(angle)
                time.sleep(scan_speed)
                
                distance = px.ultrasonic.read()
                if distance is not None and distance > 0:
                    readings.append((distance, angle))
                    obstacle_map.update_obstacle(distance, angle)
                
                if len(readings) % 5 == 0:
                    obstacle_map.interpolate_readings(readings[-10:])
                    print_maps(obstacle_map)
            
            # Scan back from max to min angle
            for angle in range(obstacle_map.max_angle, obstacle_map.min_angle - 1, -step):
                px.set_cam_pan_angle(angle)
                time.sleep(scan_speed)
                
                distance = px.ultrasonic.read()
                if distance is not None and distance > 0:
                    readings.append((distance, angle))
                    obstacle_map.update_obstacle(distance, angle)
                
                if len(readings) % 5 == 0:
                    obstacle_map.interpolate_readings(readings[-10:])
                    print_maps(obstacle_map)
            
            # Keep only recent readings
            if len(readings) > 1000:
                readings = readings[-1000:]
                
    except KeyboardInterrupt:
        return readings
    
    return readings

def print_maps(obstacle_map: ObstacleMap):
    """
    Print both the polar and 2D grid visualizations of the obstacle map.
    
    Args:
        obstacle_map: ObstacleMap instance to visualize
    """
    os.system('clear' if os.name == 'posix' else 'cls')
    
    # Print polar map
    print("\nPolar View (distance at each angle):")
    print("-" * 60)
    
    # Create angle labels
    angles = list(range(obstacle_map.min_angle, obstacle_map.max_angle + 1, 15))
    angle_positions = [int((angle - obstacle_map.min_angle) * 58 / obstacle_map.angle_range) for angle in angles]
    
    # Print angle scale
    print(" " * 5, end="")
    for i in range(58):
        found = False
        for j, pos in enumerate(angle_positions):
            if i == pos:
                print(f"{angles[j]:3d}", end="")
                found = True
                break
        if not found:
            print(" ", end="")
    print("\nAngle:", end="")
    
    # Print distance markers
    for i in range(58):
        idx = int(i * obstacle_map.angle_range / 58)
        if idx < len(obstacle_map.polar_map):
            dist = obstacle_map.polar_map[idx]
            if np.isinf(dist):
                print("·", end="")
            else:
                print("■", end="")
    print("\n" + "-" * 60)
    
    # Print 2D grid map
    print("\n2D Grid Map (■ = obstacle, · = empty, C = car):")
    print("-" * (obstacle_map.grid_size + 2))
    
    for y in range(obstacle_map.grid_size):
        print("|", end="")
        for x in range(obstacle_map.grid_size):
            if x == int(obstacle_map.car_pos[0]) and y == int(obstacle_map.car_pos[1]):
                print("C", end="")
            elif obstacle_map.grid[y, x] == 1:
                print("■", end="")
            else:
                print("·", end="")
        print("|")
    
    print("-" * (obstacle_map.grid_size + 2))
    print(f"Resolution: {obstacle_map.resolution}cm per cell")

if __name__ == "__main__":
    try:
        px = Picarx()
        obstacle_map = ObstacleMap(size=100, resolution=2.0, angle_range=(-90, 90))
        readings = scan_environment(px, obstacle_map)
           
    except KeyboardInterrupt:
        print("\nMapping stopped by user")
    finally:
        px.set_cam_pan_angle(0)
        time.sleep(0.5)