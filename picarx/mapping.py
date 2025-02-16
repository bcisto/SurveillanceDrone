import numpy as np
from typing import Tuple, List
import math


class Map2D:
    def __init__(self, size: int = 100, resolution: float = 1.0):
        """
        Initialize a 2D map for environment representation.
       
        Args:
            size: Size of the square grid (size x size)
            resolution: Resolution in cm per grid cell
        """
        self.size = size
        self.resolution = resolution
        self.grid = np.zeros((size, size), dtype=np.uint8)
        self.position = np.array([size//2, size//2])  # Start at center
        self.heading = 0  # Heading in radians (0 is pointing right/east)
       
    def update_position(self, dx: float, dy: float) -> None:
        """
        Update the car's position based on movement.
       
        Args:
            dx: Change in x position (cm)
            dy: Change in y position (cm)
        """
        # Convert to grid coordinates
        grid_dx = dx / self.resolution
        grid_dy = dy / self.resolution
       
        # Update position
        self.position += np.array([grid_dx, grid_dy])
        # Ensure position stays within bounds
        self.position = np.clip(self.position, 0, self.size-1)
       
    def update_heading(self, angle_deg: float) -> None:
        """
        Update the car's heading.
       
        Args:
            angle_deg: New heading angle in degrees
        """
        self.heading = math.radians(angle_deg)
       
    def add_obstacle(self, distance: float, angle_deg: float) -> None:
        """
        Add an obstacle to the map based on sensor reading.
       
        Args:
            distance: Distance to obstacle in cm
            angle_deg: Angle of the sensor reading in degrees
        """
        if distance <= 0:
            return
           
        # Convert to radians
        angle_rad = math.radians(angle_deg)
        total_angle = self.heading + angle_rad
       
        # Calculate obstacle position in grid coordinates
        grid_distance = distance / self.resolution
        dx = grid_distance * math.cos(total_angle)
        dy = grid_distance * math.sin(total_angle)
       
        obstacle_pos = self.position + np.array([dx, dy])
        x, y = int(obstacle_pos[0]), int(obstacle_pos[1])
       
        # Check bounds
        if 0 <= x < self.size and 0 <= y < self.size:
            self.grid[y, x] = 1
           
    def interpolate_obstacles(self, readings: List[Tuple[float, float]]) -> None:
        """
        Interpolate between sensor readings to fill in gaps.
       
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
               
            # Linear interpolation
            angle_steps = np.linspace(angle1, angle2, num=int(abs(angle2-angle1)*2))
            for angle in angle_steps:
                # Linear interpolation of distance
                t = (angle - angle1) / (angle2 - angle1)
                distance = dist1 + t * (dist2 - dist1)
                self.add_obstacle(distance, angle)
               
    def get_map(self) -> np.ndarray:
        """
        Get the current map representation.
       
        Returns:
            2D numpy array representing the environment
        """
        return self.grid.copy()