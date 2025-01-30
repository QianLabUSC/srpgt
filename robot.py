import pygame
import math
import numpy as np

class Robot:
    def __init__(self, x, y, radius, color=(255,255,255), screen_width=800, screen_height=600, BUFFER_SIZE=0):
        self.trail = []
        self.pos = [x, y]
        self.radius = radius
        self.angle_line_length = 40
        self.angle = 0
        self.move_speed = 0.02
        self.rotation_speed = 0.04
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color = color
        self.BUFFER_SIZE = BUFFER_SIZE

    def update(self, keys, robot_pos_transformed,projected_goal):
        # Add current position to trail
        self.trail.append(self.pos.copy())
        # Calculate direction vector to the projected goal
        direction = projected_goal - robot_pos_transformed
        distance_to_goal = np.linalg.norm(direction)
        if distance_to_goal > 1:  # Only move if the goal is not already reached
            direction = direction / distance_to_goal
            self.pos += self.move_speed*distance_to_goal * direction
            
        self.angle = math.atan2(direction[1], direction[0])

        # Ensure the robot stays within screen boundaries
        self.pos[0] = max(self.radius, min(self.screen_width - self.radius, self.pos[0]))
        self.pos[1] = max(self.radius, min(self.screen_height - self.radius, self.pos[1]))
    def draw(self, screen):
        # Draw the trail
        if len(self.trail) > 1:
            pygame.draw.lines(screen, (0,0,255), False, [(int(pos[0])*self.BUFFER_SIZE, int(pos[1])*self.BUFFER_SIZE) for pos in self.trail], 2)
        # Draw the robot
        pygame.draw.circle(screen, self.color, (int(self.pos[0])*self.BUFFER_SIZE, int(self.pos[1])*self.BUFFER_SIZE), self.radius*self.BUFFER_SIZE)
        # Draw the direction line
        pygame.draw.line(screen, self.color, (int(self.pos[0])*self.BUFFER_SIZE, int(self.pos[1])*self.BUFFER_SIZE), (
            int(self.pos[0]*self.BUFFER_SIZE + self.angle_line_length * math.cos(self.angle)),
            int(self.pos[1]*self.BUFFER_SIZE + self.angle_line_length * math.sin(self.angle))
        ))