import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import math
import numpy as np
from robot import Robot
from obstacle import Obstacle
from scipy.spatial import Voronoi, ConvexHull
import shapely as sp
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.geometry import LineString
from concave_hull import concave_hull, concave_hull_indexes
from disjoint import build_disjoint_sets
from reactive_planner_lib import diffeoTreeTriangulation, polygonDiffeoTriangulation
import visualization
import time

from laguerre_voronoi_2d.laguerre_voronoi_2d import get_power_triangulation, get_voronoi_cells

FILENAME = 'testvalues.csv'
Y = np.loadtxt(FILENAME, delimiter=',', skiprows=1)

Y_shape = Y.shape

# Initialize Pygame
pygame.init()

BUFFER_SIZE = 1

# Set up the display
screen_width = Y_shape[1] * BUFFER_SIZE
screen_height = Y_shape[0] * BUFFER_SIZE
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

ROBOT_RADIUS = 10

# Instantiate the robot
robot = Robot(10, 200, ROBOT_RADIUS, BLACK, screen_width, screen_height, BUFFER_SIZE=BUFFER_SIZE)

# Instantiate the goal point
goal = np.array([700, 100])

parameter_set = []

for i in range(Y_shape[0]):
    for j in range(Y_shape[1]):
        parameter_set.append((i, j))

THRESHOLD = 0

parameter_set_filtered = []

for i, j in parameter_set:
    if Y[i, j] < THRESHOLD:
        parameter_set_filtered.append((i, j))
        
        
# disjoint_sets = build_disjoint_sets(parameter_set_filtered, 1)


# create concave hulls from disjoint sets

concave_obstacles = [
    # Concave "C" shape
    np.array([[100, 200], [200, 200], [200, 300], [150, 300], [150, 250], [100, 250], [100, 200]]),
    
    # Concave "L" shape
    np.array([[350, 100], [450, 100], [450, 150], [400, 150], [400, 300], [350, 300], [350, 100]]),

    # Concave "U" shape
    np.array([[250, 400], [350, 400], [350, 450], [300, 450], [300, 500], [250, 500], [250, 400]]),

    # # Concave "Arrow" shape
    # np.array([[500, 300], [550, 350], [520, 350], [520, 400], [480, 400], [480, 350], [450, 350], [500, 300]])
]

obstacles = []

polygon_list = []

# for i, group in enumerate(disjoint_sets):
#     if len(group) > 2:
#         points = [parameter_set_filtered[i] for i in group]
#         points = np.array(points)
#         if len(group) > 3:
#             hull = concave_hull(points, concavity=2, length_threshold=0)
#             poly = Polygon(hull)
#         else:
#             poly = Polygon(points) 
#         polygon_list.append(poly.buffer(ROBOT_RADIUS, join_style=2))
#         # print new poly vertices count
#         # print (len(poly.exterior.coords))

for concave_obstacle in concave_obstacles:
    poly = Polygon(concave_obstacle)
    polygon_list.append(poly.buffer(ROBOT_RADIUS, join_style=2))
        

diffeo_params = dict()
diffeo_params['p'] = 50
diffeo_params['epsilon'] = 50
diffeo_params['varepsilon'] = 50
diffeo_params['mu_1'] = 50
diffeo_params['mu_2'] = 0.1
diffeo_params['workspace'] = np.array([[0,0],[Y_shape[0],0],[Y_shape[0],Y_shape[1]],[0,Y_shape[1]],[0,0]])

# visualization.visualize_diffeoDeterminant_triangulation(polygon_list, 10, np.array([0, Y_shape[0], 0, Y_shape[1]]), np.array([101,101]), diffeo_params)

# check for polygons in range of each other + robot radius

polygon_list_merged = []
i = 0
while (i<len(polygon_list)):
    polygon_list_merged.append(polygon_list[i])

    j = i+1
    while (j<len(polygon_list)):
        if polygon_list_merged[i].intersects(polygon_list[j]):
            polygon_list_merged[i] = polygon_list_merged[i].union(polygon_list[j])
            polygon_list_merged[i] = polygon_list_merged[i].simplify(0.08, preserve_topology=True) # simplify polygon to eliminate strange small corners
            del(polygon_list[j])
        else:
            j = j+1
    polygon_list_merged[i] = sp.geometry.polygon.orient(polygon_list_merged[i], 1.0) # orient polygon to be CCW
    i = i+1
print ("merged polygon list length: ", len(polygon_list_merged))

# visualization.visualize_diffeoDeterminant_triangulation(polygon_list_merged, 10, np.array([0, Y_shape[0], 0, Y_shape[1]]), np.array([101,101]), diffeo_params)
diffeo_tree_array = []
for i in range(len(polygon_list_merged)):
    coords = np.vstack((polygon_list_merged[i].exterior.coords.xy[0],polygon_list_merged[i].exterior.coords.xy[1])).transpose()
    print("i: ", i)
    if i == -1:
        continue
    diffeo_tree_array.append(diffeoTreeTriangulation(coords, diffeo_params))


print ("Diffeo tree array length: ", len(diffeo_tree_array))

for tree in diffeo_tree_array:
    radius = tree[-1]['radius']
    center = tree[-1]['center'].squeeze()
    obstacles.append(Obstacle(center[0], center[1], radius, BLACK, BUFFER_SIZE))

def transform_point(point, diffeo_tree_array, diffeo_params):
    PositionTransformed = np.array([[point[0],point[1]]])
    PositionTransformedD = np.eye(2)
    PositionTransformedDD = np.zeros(8)
    for k in range(len(diffeo_tree_array)):
        TempPositionTransformed, TempPositionTransformedD, TempPositionTransformedDD = polygonDiffeoTriangulation(PositionTransformed, diffeo_tree_array[k], diffeo_params)

        res1 = TempPositionTransformedD[0][0]*PositionTransformedDD[0] + TempPositionTransformedD[0][1]*PositionTransformedDD[4] + PositionTransformedD[0][0]*(TempPositionTransformedDD[0]*PositionTransformedD[0][0] + TempPositionTransformedDD[1]*PositionTransformedD[1][0]) + PositionTransformedD[1][0]*(TempPositionTransformedDD[2]*PositionTransformedD[0][0] + TempPositionTransformedDD[3]*PositionTransformedD[1][0])
        res2 = TempPositionTransformedD[0][0]*PositionTransformedDD[1] + TempPositionTransformedD[0][1]*PositionTransformedDD[5] + PositionTransformedD[0][0]*(TempPositionTransformedDD[0]*PositionTransformedD[0][1] + TempPositionTransformedDD[1]*PositionTransformedD[1][1]) + PositionTransformedD[1][0]*(TempPositionTransformedDD[2]*PositionTransformedD[0][1] + TempPositionTransformedDD[3]*PositionTransformedD[1][1])
        res3 = TempPositionTransformedD[0][0]*PositionTransformedDD[2] + TempPositionTransformedD[0][1]*PositionTransformedDD[6] + PositionTransformedD[0][1]*(TempPositionTransformedDD[0]*PositionTransformedD[0][0] + TempPositionTransformedDD[1]*PositionTransformedD[1][0]) + PositionTransformedD[1][1]*(TempPositionTransformedDD[2]*PositionTransformedD[0][0] + TempPositionTransformedDD[3]*PositionTransformedD[1][0])
        res4 = TempPositionTransformedD[0][0]*PositionTransformedDD[3] + TempPositionTransformedD[0][1]*PositionTransformedDD[7] + PositionTransformedD[0][1]*(TempPositionTransformedDD[0]*PositionTransformedD[0][1] + TempPositionTransformedDD[1]*PositionTransformedD[1][1]) + PositionTransformedD[1][1]*(TempPositionTransformedDD[2]*PositionTransformedD[0][1] + TempPositionTransformedDD[3]*PositionTransformedD[1][1])
        res5 = TempPositionTransformedD[1][0]*PositionTransformedDD[0] + TempPositionTransformedD[1][1]*PositionTransformedDD[4] + PositionTransformedD[0][0]*(TempPositionTransformedDD[4]*PositionTransformedD[0][0] + TempPositionTransformedDD[5]*PositionTransformedD[1][0]) + PositionTransformedD[1][0]*(TempPositionTransformedDD[6]*PositionTransformedD[0][0] + TempPositionTransformedDD[7]*PositionTransformedD[1][0])
        res6 = TempPositionTransformedD[1][0]*PositionTransformedDD[1] + TempPositionTransformedD[1][1]*PositionTransformedDD[5] + PositionTransformedD[0][0]*(TempPositionTransformedDD[4]*PositionTransformedD[0][1] + TempPositionTransformedDD[5]*PositionTransformedD[1][1]) + PositionTransformedD[1][0]*(TempPositionTransformedDD[6]*PositionTransformedD[0][1] + TempPositionTransformedDD[7]*PositionTransformedD[1][1])
        res7 = TempPositionTransformedD[1][0]*PositionTransformedDD[2] + TempPositionTransformedD[1][1]*PositionTransformedDD[6] + PositionTransformedD[0][1]*(TempPositionTransformedDD[4]*PositionTransformedD[0][0] + TempPositionTransformedDD[5]*PositionTransformedD[1][0]) + PositionTransformedD[1][1]*(TempPositionTransformedDD[6]*PositionTransformedD[0][0] + TempPositionTransformedDD[7]*PositionTransformedD[1][0])
        res8 = TempPositionTransformedD[1][0]*PositionTransformedDD[3] + TempPositionTransformedD[1][1]*PositionTransformedDD[7] + PositionTransformedD[0][1]*(TempPositionTransformedDD[4]*PositionTransformedD[0][1] + TempPositionTransformedDD[5]*PositionTransformedD[1][1]) + PositionTransformedD[1][1]*(TempPositionTransformedDD[6]*PositionTransformedD[0][1] + TempPositionTransformedDD[7]*PositionTransformedD[1][1])
        PositionTransformedDD[0] = res1
        PositionTransformedDD[1] = res2
        PositionTransformedDD[2] = res3
        PositionTransformedDD[3] = res4
        PositionTransformedDD[4] = res5
        PositionTransformedDD[5] = res6
        PositionTransformedDD[6] = res7
        PositionTransformedDD[7] = res8

        PositionTransformedD = np.matmul(TempPositionTransformedD, PositionTransformedD)

        PositionTransformed = TempPositionTransformed
    
    return PositionTransformed, PositionTransformedD, PositionTransformedDD

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
                pygame.draw.line(screen, GREEN, (int(start_point[0])*BUFFER_SIZE, int(start_point[1])*BUFFER_SIZE), (int(end_point[0])*BUFFER_SIZE, int(end_point[1])*BUFFER_SIZE), 1)

def compute_local_workspace_polygon(robot, obstacles):

    # Gather the positions of the robot and obstacles
    
    robot_pos_transformed, _, _ = transform_point(robot.pos, diffeo_tree_array, diffeo_params)
    robot_pos_transformed = robot_pos_transformed.squeeze()
    
    points = [robot_pos_transformed] + [obstacle.pos for obstacle in obstacles]
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
        pygame_points = [(int(x)*BUFFER_SIZE, int(y)*BUFFER_SIZE) for x, y in polygon.exterior.coords]
            
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
        pygame_points = [(int(x)*BUFFER_SIZE, int(y)*BUFFER_SIZE) for x, y in polygon.exterior.coords]
        pygame.draw.polygon(screen, PINK, pygame_points)

clock = pygame.time.Clock()

keys = pygame.key.get_pressed()

# Game loop
running = True
update_robot = False

# Get max and min values of Y

max_Y = np.max(Y)
min_Y = np.min(Y)
    
Z = 255*(Y - min_Y) / (max_Y - min_Y)
Z = np.repeat(Z, BUFFER_SIZE, axis=1)
Z = np.repeat(Z, BUFFER_SIZE, axis=0).T
surf = pygame.surfarray.make_surface(Z)


# Draw the goal point
def draw_goal(screen, goal):
    pygame.draw.circle(screen, BLUE, (int(goal[0])*BUFFER_SIZE, int(goal[1])*BUFFER_SIZE), 10*BUFFER_SIZE)

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                goal = np.array(pygame.mouse.get_pos())/BUFFER_SIZE

    # Handle key inputs
    keys = pygame.key.get_pressed()

    # Clear the screen
    screen.fill(WHITE)
    
    # draw squares according to ground truth values
    
    # screen.blit(surf, (0, 0))
    
    # for i, j in parameter_set_filtered:
    #     pygame.draw.rect(screen, (0, 0, 0), (j * BUFFER_SIZE, i * BUFFER_SIZE, BUFFER_SIZE, BUFFER_SIZE))

    robot_pos_transformed, _, _ = transform_point(robot.pos, diffeo_tree_array, diffeo_params)
    robot_pos_transformed = robot_pos_transformed.squeeze()

    
    local_workspace_polygon = compute_local_workspace_polygon(robot, obstacles)

    # # Fill the local workspace
    draw_local_workspace_polygon(screen, local_workspace_polygon)
    
    local_free_space_polygon = compute_local_free_space_polygon(local_workspace_polygon, robot)
    # local_free_space_polygon = local_workspace_polygon
    
    draw_local_free_space_polygon(screen, local_free_space_polygon)
    for concave_hull in polygon_list_merged:
        if concave_hull is not None:
            pygame_points = [(int(x)*BUFFER_SIZE, int(y)*BUFFER_SIZE) for x, y in concave_hull.exterior.coords]
            pygame.draw.polygon(screen, GREEN, pygame_points)
    # for polygon in transformed_polygons:
    #     if polygon is not None:
    #         pygame_points = [(int(x)*BUFFER_SIZE, int(y)*BUFFER_SIZE) for y, x in polygon]
    #         for point in pygame_points:
    #             pygame.draw.circle(screen, RED, point, 5)
            
            
    
    # Draw the obstacles
    for obstacle in obstacles:
        obstacle.draw(screen)
        
    # Draw the Voronoi diagram
    draw_power_diagram(screen, obstacles, robot)
    
    # Draw the robot
    robot.draw(screen)

    # draw robot transformed
    
    pygame.draw.circle(screen, BLACK, (int(robot_pos_transformed[0])*BUFFER_SIZE, int(robot_pos_transformed[1])*BUFFER_SIZE), ROBOT_RADIUS*BUFFER_SIZE)

    transformed_mouse_pos, _, _ = transform_point(pygame.mouse.get_pos(), diffeo_tree_array, diffeo_params)
    transformed_mouse_pos = transformed_mouse_pos.squeeze()

    pygame.draw.circle(screen, BLACK, (int(transformed_mouse_pos[0])*BUFFER_SIZE, int(transformed_mouse_pos[1])*BUFFER_SIZE), 5*BUFFER_SIZE)

    # Project the goal to the edge of the polygon
    projected_goal = project_goal_to_polygon(goal, local_free_space_polygon)
    
    if update_robot:
        robot.update(keys, robot_pos_transformed.squeeze(), projected_goal)
        pass
        
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
