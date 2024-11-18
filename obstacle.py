import pygame
import numpy as np

class Obstacle:
    
    def __init__(self, x, y, radius=10, color=(255, 0, 0)):
        self.pos = np.array([x, y])
        self.radius = radius
        self.color = color

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), self.radius)