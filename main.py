import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import numpy as np
from robot import Robot
from obstacle import Obstacle
from scipy.spatial import Voronoi, ConvexHull
import shapely as sp
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.geometry import LineString
from concave_hull import concave_hull, concave_hull_indexes
from disjoint import build_disjoint_sets
import GPy
from safeopt import SafeOpt, linearly_spaced_combinations

FILENAME = 'testvalues.csv'
Y = np.loadtxt(FILENAME, delimiter=',', skiprows=1)

Y_shape = Y.shape

# Initialize Pygame
pygame.init()

BUFFER_SIZE = 1

# Set up the display
screen_width = Y_shape[0] * BUFFER_SIZE
screen_height = Y_shape[1] * BUFFER_SIZE
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
robot = Robot(100, 200, ROBOT_RADIUS, BLACK, screen_width, screen_height, BUFFER_SIZE=BUFFER_SIZE)

# Instantiate the goal point
goal = np.array([700, 100])

parameter_set = np.array([[i, j] for i in range(Y_shape[0]) for j in range(Y_shape[1])])

print ("x range: ", Y_shape[0])
print ("y range: ", Y_shape[1])

THRESHOLD = 0

start_X = [robot.pos]
start_Y = [Y[round(robot.pos[0]), round(robot.pos[1])]]

# pick a few more random points around the robot

for i in range(10):
    x = np.random.randint(max(0, robot.pos[0]-10), min(Y_shape[0], robot.pos[0]+10))
    y = np.random.randint(max(0, robot.pos[1]-10), min(Y_shape[1], robot.pos[1]+10))
    start_X.append([x, y])
    start_Y.append(Y[x, y])
    
starting_X = np.array(start_X)
starting_Y = np.array(start_Y)
starting_Y = starting_Y.reshape(-1, 1)



KERNEL_VARIANCE = 20
KERNEL_LENGTHSCALE = 50.0
LIPSCHITZ_CONSTANT = None
BETA = 2.0

kernel = GPy.kern.RBF(input_dim=2, variance=KERNEL_VARIANCE, lengthscale=KERNEL_LENGTHSCALE)
gp = GPy.models.GPRegression(starting_X, starting_Y, kernel)
opt = SafeOpt(gp, parameter_set, THRESHOLD, LIPSCHITZ_CONSTANT, BETA)
next_parameters = opt.optimize()
    



parameter_set_filtered = []

for index, value in enumerate(opt.S):
    if value:
        parameter_set_filtered.append(parameter_set[index])

# print("next_parameters: ", parameter_set_filtered)
        
disjoint_sets = build_disjoint_sets(parameter_set_filtered, 1)



obstacles = []

polygon_list = []
polygon_list_original = []

SIMPLIFICATION_CONSTANT = 20

for i, group in enumerate(disjoint_sets):
    if len(group) > 2:
        points = [parameter_set_filtered[i] for i in group]
        points = np.array(points)
        if len(group) > 3:
            hull = concave_hull(points, concavity=2, length_threshold=0)
            poly = Polygon(hull)
        else:
            poly = Polygon(points) 
        polygon_list.append(poly.buffer(ROBOT_RADIUS, join_style=2))
        polygon_list_original.append(poly)
#         # print new poly vertices count
#         # print (len(poly.exterior.coords))

# for concave_obstacle in concave_obstacles:
#     poly = Polygon(concave_obstacle)
#     polygon_list.append(poly.buffer(ROBOT_RADIUS, join_style=2))
#     polygon_list_original.append(poly)

# check for polygons in range of each other + robot radius

polygon_list_merged = []
i = 0
while (i<len(polygon_list)):
    polygon_list_merged.append(polygon_list[i])

    j = i+1
    while (j<len(polygon_list)):
        if polygon_list_merged[i].intersects(polygon_list[j]):
            polygon_list_merged[i] = polygon_list_merged[i].union(polygon_list[j])
            print(polygon_list_merged[i])
            polygon_list_merged[i] = polygon_list_merged[i].simplify(10, preserve_topology=True) # simplify polygon to eliminate strange small corners
            del(polygon_list[j])
        else:
            j = j+1
    polygon_list_merged[i] = sp.geometry.polygon.orient(polygon_list_merged[i], 1.0) # orient polygon to be CCW
    i = i+1
print ("merged polygon list length: ", len(polygon_list_merged))

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
Z = np.repeat(Z, BUFFER_SIZE, axis=0)

# Z is a surface

surf = pygame.surfarray.make_surface(Z)

# make safe surface starting with Z 

surf_safe = np.zeros((Y_shape[0]*BUFFER_SIZE, Y_shape[1]*BUFFER_SIZE, 3), dtype=np.uint8)
surf_safe[:, :, 0] = Z
surf_safe[:, :, 1] = Z
surf_safe[:, :, 2] = Z
surf_safe = np.repeat(surf_safe, BUFFER_SIZE, axis=1)
surf_safe = np.repeat(surf_safe, BUFFER_SIZE, axis=0)

for i, j in parameter_set_filtered:
    surf_safe[int(i*BUFFER_SIZE):int((i+1)*BUFFER_SIZE), int(j*BUFFER_SIZE):int((j+1)*BUFFER_SIZE), :] = LIGHT_BLUE



surf_safe_surface = pygame.surfarray.make_surface(surf_safe)


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
    
    screen.blit(surf_safe_surface, (0, 0))
    
    # for i, j in parameter_set_filtered:
    #     pygame.draw.rect(screen, (0, 0, 0), (j * BUFFER_SIZE, i * BUFFER_SIZE, BUFFER_SIZE, BUFFER_SIZE))
    
    # for polygon in polygon_list_merged:
    #     x, y = polygon.exterior.xy
    #     pygame.draw.polygon(screen, LIGHT_BLUE, np.array([x, y]).T * BUFFER_SIZE)
    
    # draw next parameters
    pygame.draw.circle(screen, PINK, (int(next_parameters[0])*BUFFER_SIZE, int(next_parameters[1])*BUFFER_SIZE), 10*BUFFER_SIZE)
    
    if update_robot:
        u = next_parameters - robot.pos
        
        # implement A*
        
        
        robot.update(u)
        if np.linalg.norm(u) < 1:
            opt.add_new_data_point(robot.pos, Y[round(robot.pos[0]), round(robot.pos[1])])
            opt.optimize()
            next_parameters = opt.optimize()
            #update safe set
            parameter_set_filtered = []

            for index, value in enumerate(opt.S):
                if value:
                    parameter_set_filtered.append(parameter_set[index])
                    
            for i, j in parameter_set_filtered:
                surf_safe[int(i*BUFFER_SIZE):int((i+1)*BUFFER_SIZE), int(j*BUFFER_SIZE):int((j+1)*BUFFER_SIZE), :] = LIGHT_BLUE
            surf_safe_surface = pygame.surfarray.make_surface(surf_safe)
            
    
        
        
    if keys[pygame.K_SPACE] and not last_keys[pygame.K_SPACE]: # get only rising edge
        update_robot = not update_robot
        
    last_keys = keys

    robot.draw(screen)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit the game
pygame.quit()
