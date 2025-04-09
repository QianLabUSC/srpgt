import pygame
import numpy as np
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pure Pursuit Controller Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

class PurePursuitController:
    def __init__(self, lookahead_distance=50.0):
        """
        Initialize the Pure Pursuit controller
        
        Args:
            lookahead_distance: Distance to look ahead on the path (pixels)
        """
        self.lookahead_distance = lookahead_distance
        
    def find_target_point(self, robot_position, path):
        """
        Find the target point on the path that is lookahead_distance away
        
        Args:
            robot_position: Current position of the robot [x, y]
            path: List of waypoints [[x1, y1], [x2, y2], ...]
            
        Returns:
            target_point: The target point to pursue [x, y]
            target_index: Index of the closest point on the path
        """
        # Convert to numpy arrays for easier calculations
        robot_position = np.array(robot_position)
        path = np.array(path)
        
        # Find distances from robot to all points on the path
        distances = np.linalg.norm(path - robot_position, axis=1)
        
        # Find index of closest point
        closest_index = np.argmin(distances)
        
        # Search for the point that is approximately lookahead_distance away
        target_index = closest_index
        for i in range(closest_index, len(path)):
            target_index = i
            # If we've gone far enough ahead on the path
            if np.linalg.norm(path[i] - robot_position) >= self.lookahead_distance:
                break
                
        # If we're near the end of the path, use the last point
        if target_index >= len(path) - 1:
            target_index = len(path) - 1
            
        return path[target_index], target_index
    
    def compute_velocity(self, robot_position, target_point, max_speed=100.0):
        """
        Compute the velocity command to reach the target point
        
        Args:
            robot_position: Current position of the robot [x, y]
            target_point: Target point to pursue [x, y]
            max_speed: Maximum speed of the robot (pixels/second)
            
        Returns:
            velocity_command: Velocity command to send to the robot [vx, vy]
        """
        # Convert to numpy arrays
        robot_position = np.array(robot_position)
        target_point = np.array(target_point)
        
        # Vector from robot to target point
        direction = target_point - robot_position
        
        # Normalize the direction vector and scale by max_speed
        distance = np.linalg.norm(direction)
        if distance > 0:
            velocity_command = (direction / distance) * max_speed
        else:
            velocity_command = np.zeros(2)
            
        return velocity_command

def main():
    # Clock for controlling simulation speed
    clock = pygame.time.Clock()
    FPS = 60
    
    # Create a path (a simple figure-8 in this case)
    t = np.linspace(0, 2*np.pi, 200)
    path = np.array([200*np.sin(t) + WIDTH//2, 150*np.sin(2*t) + HEIGHT//2]).T
    
    # Initialize robot
    robot_position = np.array([path[0][0], path[0][1]], dtype=float)
    robot_velocity = np.zeros(2)
    robot_radius = 10
    robot_trail = []
    max_trail_length = 100
    
    # Initialize controller
    controller = PurePursuitController(lookahead_distance=80.0)
    
    # Simulation parameters
    dt = 1.0 / FPS  # time step
    max_speed = 150.0  # maximum speed (pixels/second)
    
    # Main game loop
    running = True
    paused = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    controller.lookahead_distance += 10
                elif event.key == pygame.K_DOWN:
                    controller.lookahead_distance = max(10, controller.lookahead_distance - 10)
        
        if not paused:
            # Find target point
            target_point, target_index = controller.find_target_point(robot_position, path)
            
            # Compute velocity command
            velocity_command = controller.compute_velocity(robot_position, target_point, max_speed)
            
            # Update robot position
            robot_position = robot_position + velocity_command * dt
            robot_velocity = velocity_command
            
            # Add position to trail
            robot_trail.append(tuple(robot_position))
            if len(robot_trail) > max_trail_length:
                robot_trail.pop(0)
            
            # Check if we've reached the end of the path
            if target_index >= len(path) - 1 and np.linalg.norm(robot_position - path[-1]) < 5:
                # Reset to beginning of path
                robot_position = np.array([path[0][0], path[0][1]], dtype=float)
                robot_trail = []
        
        # Draw everything
        screen.fill(WHITE)
        
        # Draw path
        if len(path) > 1:
            pygame.draw.lines(screen, BLUE, False, path, 2)
        
        # Draw robot trail
        if len(robot_trail) > 1:
            pygame.draw.lines(screen, (200, 200, 200), False, robot_trail, 2)
        
        # Draw target point
        pygame.draw.circle(screen, GREEN, target_point.astype(int), 5)
        
        # Draw line from robot to target
        pygame.draw.line(screen, YELLOW, robot_position.astype(int), target_point.astype(int), 2)
        
        # Draw robot (as a circle)
        pygame.draw.circle(screen, RED, robot_position.astype(int), robot_radius)
        
        # Draw lookahead distance circle
        pygame.draw.circle(screen, (200, 200, 200), robot_position.astype(int), 
                        controller.lookahead_distance, 1)
        
        # Display information
        font = pygame.font.Font(None, 24)
        info_text = font.render(f"Lookahead: {controller.lookahead_distance:.1f} px  |  Speed: {np.linalg.norm(robot_velocity):.1f} px/s", 
                             True, BLACK)
        screen.blit(info_text, (10, 10))
        
        controls_text = font.render("SPACE: Pause  |  UP/DOWN: Adjust lookahead distance", True, BLACK)
        screen.blit(controls_text, (10, HEIGHT - 30))
        
        # Update the display
        pygame.display.flip()
        
        # Control simulation speed
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()