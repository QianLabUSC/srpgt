import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import math
import numpy as np
from robot import Robot
from obstacle import Obstacle
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
from shapely.geometry import LineString
import time

from laguerre_voronoi_2d.laguerre_voronoi_2d import get_power_triangulation, get_voronoi_cells

# Initialize Pygame
pygame.init()

# Set up the display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Disk Robot Simulator")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
LIGHT_BLUE = (173, 216, 230)
BLUE = (0, 0, 255)
PINK = (255, 105, 180)

# Instantiate the robot
robot = Robot(100, 500, 10, BLACK, screen_width, screen_height)

# Instantiate the goal point
goal = np.array([700, 100])

# Instantiate obstacles 
obstacles = [
    Obstacle(100, 150, 25),
    Obstacle(300, 400, 20),
    Obstacle(500, 200, 70),
    Obstacle(700, 500, 30),
    Obstacle(400, 100, 70),
]

def draw_power_diagram(screen, points, radii):

    # Gather the positions of the robot and obstacles
    points = [robot.pos] + [obstacle.pos for obstacle in obstacles]
    radii = [robot.radius] + [obstacle.radius for obstacle in obstacles]
    
    
    tri_list, V = get_power_triangulation(np.array(points), np.array(radii))
    
    # Get the Voronoi cells from the power triangulation
    voronoi_cell_map = get_voronoi_cells(np.array(points), V, tri_list)
    
    # Draw each Voronoi cell
    for segments in voronoi_cell_map.values():
        for segment in segments:
            (_, _), (A, U, tmin, tmax) = segment
            if tmin is None:
                tmin = -screen_width - screen_height
            if tmax is None:
                tmax = screen_width + screen_height
            if tmin is not None and tmax is not None:
                start_point = A + tmin * U
                end_point = A + tmax * U
                pygame.draw.line(screen, GREEN, (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])), 1)

def compute_local_workspace_polygon(robot, obstacles):

    # Gather the positions of the robot and obstacles
    points = [robot.pos] + [obstacle.pos for obstacle in obstacles]
    radii = [robot.radius] + [obstacle.radius for obstacle in obstacles]

    # Get the power triangulation (Laguerre triangulation)
    tri_list, V = get_power_triangulation(np.array(points), np.array(radii))
    
    # Get the Voronoi cells from the power triangulation
    voronoi_cell_map = get_voronoi_cells(np.array(points), V, tri_list)
    
    # Extract the robot's Voronoi cell
    robot_voronoi_segments = voronoi_cell_map[0]
    
    # Collect the vertices from the segments
    vertices = []
    for segment in robot_voronoi_segments:
        (_, _), (A, U, tmin, tmax) = segment
        if tmin is not None and tmax is not None:
            vertices.append(A + tmin * U)
            vertices.append(A + tmax * U)
        elif tmin is not None:  # Infinite line segments
            vertices.append(A + tmin * U)
            vertices.append(A + (screen_height + screen_width) * U)
        elif tmax is not None:
            vertices.append(A + tmax * U)
            vertices.append(A - (screen_height + screen_width) * U)

    # Remove duplicate vertices
    vertices = list({tuple(v) for v in vertices})

    # Create a convex hull polygon from the vertices
    if len(vertices) > 2:
        hull = Polygon(vertices).convex_hull
        return hull
    return None


def project_goal_to_polygon(goal, polygon):
    if polygon is not None:
        goal_point = Point(goal)
        if polygon.contains(goal_point):
            return goal  # If the goal is inside the polygon, no projection needed
        projected_point = polygon.boundary.interpolate(polygon.boundary.project(goal_point))
        return np.array([projected_point.x, projected_point.y])
    return goal

def draw_local_workspace_polygon(screen, polygon):
    if polygon is not None:
        pygame_points = [(int(x), int(y)) for x, y in polygon.exterior.coords]
        pygame.draw.polygon(screen, LIGHT_BLUE, pygame_points)

def compute_local_free_space_polygon(local_workspace, robot):
    if local_workspace is not None:
        # Erode the workspace by the robot's radius to get the local free space
        local_free_space = local_workspace.buffer(-robot.radius)
        if not local_free_space.is_empty:
            return local_free_space
    return None

def draw_local_free_space_polygon(screen, polygon):
    if polygon is not None:
        pygame_points = [(int(x), int(y)) for x, y in polygon.exterior.coords]
        pygame.draw.polygon(screen, PINK, pygame_points)

clock = pygame.time.Clock()

keys = pygame.key.get_pressed()

# Game loop
running = True
update_robot = False

# Draw the goal point
def draw_goal(screen, goal):
    pygame.draw.circle(screen, BLUE, (int(goal[0]), int(goal[1])), 10)

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                goal = np.array(pygame.mouse.get_pos())

    # Handle key inputs
    keys = pygame.key.get_pressed()

    # Clear the screen
    screen.fill(WHITE)

    local_workspace_polygon = compute_local_workspace_polygon(robot, obstacles)

    # Fill the local workspace
    draw_local_workspace_polygon(screen, local_workspace_polygon)
    
    local_free_space_polygon = compute_local_free_space_polygon(local_workspace_polygon, robot)
    
    draw_local_free_space_polygon(screen, local_free_space_polygon)
    

    # Draw the obstacles
    for obstacle in obstacles:
        obstacle.draw(screen)
        
    # Draw the Voronoi diagram
    draw_power_diagram(screen, obstacles, robot)
    
    # Draw the robot
    robot.draw(screen)

    # Project the goal to the edge of the polygon
    projected_goal = project_goal_to_polygon(goal, local_free_space_polygon)
    
    if update_robot:
        robot.update(keys, projected_goal)
        
    if keys[pygame.K_SPACE] and not last_keys[pygame.K_SPACE]: # get only rising edge
        update_robot = not update_robot
        
    last_keys = keys
    
    
    # Draw the goal
    draw_goal(screen, projected_goal)
    draw_goal(screen, goal)


    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit the game
pygame.quit()
