import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# Standard libraries
import numpy as np
import tomli

# Third-party libraries
import pygame
import GPy
from scipy.spatial import ConvexHull
import shapely as sp
from shapely.geometry import Polygon, Point
from safeopt import SafeOpt
import pyvoro

# Local modules
from robot import Robot
from concave_hull import concave_hull
from disjoint import build_disjoint_sets
from reactive_planner_lib import diffeoTreeTriangulation, polygonDiffeoTriangulation

# Configuration constants - loaded from config.toml if available
CONFIG = {
    "environment": {
        "FILENAME": "testvalues.csv",
        "THRESHOLD": 0,
        "SIMPLIFICATION_CONSTANT": 3
    },
    "robot": {
        "ROBOT_RADIUS": 2
    },
    "optimization": {
        "NUM_EXPANDERS": 20,
        "KERNEL_VARIANCE": 2,
        "KERNEL_LENGTHSCALE": 5,
        "BETA": 2,
        "LIPSCHITZ": 0.1
    },
    "display": {
        "BUFFER_SIZE": 1
    }
}

def load_config(config_file="config.toml"):
    """
    Load configuration from a TOML file.
    
    Args:
        config_file: Path to the configuration file.
        
    Returns:
        Dictionary containing all configuration parameters.
    """
    try:
        with open(config_file, "rb") as f:
            loaded_config = tomli.load(f)
        
        # Update the global CONFIG with loaded values
        for section in loaded_config:
            if section in CONFIG:
                CONFIG[section].update(loaded_config[section])
            else:
                CONFIG[section] = loaded_config[section]
                
        print(f"Configuration loaded from {config_file}")
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default values.")
    except Exception as e:
        print(f"Error loading config file: {e}. Using default values.")


def get_next_parameters(opt, goal, rho):
    """
    Get the next sampling point using SafeOpt optimization.
    
    Args:
        opt: SafeOpt optimizer
        goal: Target goal position
        rho: Weight factor for distance scoring
        
    Returns:
        x: Next sampling point
    """
    l = opt.Q[:, ::2]
    u = opt.Q[:, 1::2]

    value = np.max((u[opt.G] - l[opt.G]) / opt.scaling, axis=1)
    
    # in this version of safeopt library, returned values in G are the top n closest points to the goal
    
    # so, pick the best uncertainty point

    x = opt.inputs[opt.G, :][np.argmax(value), :]
    return x

def value_to_risk(input_value):
    # return risk of input: for test purposes risk is equal to value measured. This should be changed for real world applications
    return input_value

def setup_environment(Y, Y_shape, robot, goal):
    """
    Set up the environment for robot navigation.
    
    Args:
        Y_shape: Shape of the environment map
        robot: Robot object
        goal: Target goal position
        
    Returns:
        opt: SafeOpt optimizer
        parameter_set: Array of all possible positions
        parameter_set_filtered: Array of safe positions
    """
    # Set up the environment parameters
    THRESHOLD = CONFIG["environment"]["THRESHOLD"]
    parameter_set = np.array([[i, j] for i in range(Y_shape[0]) for j in range(Y_shape[1])])
    
    # Initialize training data for Gaussian Process
    start_X = [robot.pos]
    start_Y = [Y[round(robot.pos[0]), round(robot.pos[1])]]
    
    # Add some random points around the robot for initial model
    for i in range(50):
        x = np.random.randint(max(0, robot.pos[0]-20), min(Y_shape[0], robot.pos[0]+20))
        y = np.random.randint(max(0, robot.pos[1]-20), min(Y_shape[1], robot.pos[1]+20))
        value = Y[x, y]
        if value < THRESHOLD:
            continue
        start_X.append([x, y])
        start_Y.append(value)
    
    # Convert to numpy arrays
    starting_X = np.array(start_X)
    starting_Y = np.array(start_Y).reshape(-1, 1)
    
    # Set up Gaussian Process model and SafeOpt optimizer
    KERNEL_VARIANCE = CONFIG["optimization"]["KERNEL_VARIANCE"]
    KERNEL_LENGTHSCALE = CONFIG["optimization"]["KERNEL_LENGTHSCALE"]
    BETA = CONFIG["optimization"]["BETA"]
    LIPSCHITZ = CONFIG["optimization"]["LIPSCHITZ"]
    NUM_EXPANDERS = CONFIG["optimization"]["NUM_EXPANDERS"]
    
    kernel = GPy.kern.RBF(input_dim=2, variance=KERNEL_VARIANCE, lengthscale=KERNEL_LENGTHSCALE)
    gp = GPy.models.GPRegression(starting_X, starting_Y, kernel)
    opt = SafeOpt(gp, parameter_set, fmin=THRESHOLD, lipschitz=LIPSCHITZ, beta=BETA)
    opt.update_confidence_intervals(context=None)
    # opt.compute_sets(full_sets=False, num_expanders=NUM_EXPANDERS)
    opt.compute_sets(goal=goal, num_expanders=NUM_EXPANDERS)
        
    # Get filtered safe parameters
    parameter_set_filtered = []
    for index, value in enumerate(opt.S):
        if value:
            parameter_set_filtered.append(parameter_set[index])
    
    return opt, parameter_set, parameter_set_filtered


def setup_obstacles(parameter_set_filtered, robot):
    
    SIMPLIFICATION_CONSTANT = CONFIG["environment"]["SIMPLIFICATION_CONSTANT"]
    
    # Create obstacle polygons
    polygon_list, disjoint_sets = create_obstacle_polygons(parameter_set_filtered)
    
    parameter_set_filtered = np.array(parameter_set_filtered)
    
    min_x = np.min(parameter_set_filtered[:, 0])
    max_x = np.max(parameter_set_filtered[:, 0])
    min_y = np.min(parameter_set_filtered[:, 1])
    max_y = np.max(parameter_set_filtered[:, 1])
    
    enclosing_workspace_hull_polygon = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])
    
    # Create convex hull around safe sets as enclosing workspace
    # enclosing_workspace_hull = ConvexHull(parameter_set_filtered)
    # enclosing_workspace_hull_polygon = Polygon(enclosing_workspace_hull.points[enclosing_workspace_hull.vertices])#.simplify(SIMPLIFICATION_CONSTANT, preserve_topology=True)

    # enclosing_workspace_hull_polygon = enclosing_workspace_hull_polygon.simplify(SIMPLIFICATION_CONSTANT, preserve_topology=True)
    
    
    
    
    diffeo_params = dict()
    diffeo_params['p'] = 10
    diffeo_params['epsilon'] = 4
    diffeo_params['varepsilon'] = 4
    diffeo_params['mu_1'] = 1 # beta switch parameter
    diffeo_params['mu_2'] = 0.15 # gamma switch parameter
    diffeo_params['workspace'] = np.array([list(coord) for coord in enclosing_workspace_hull_polygon.exterior.coords])
    
    # Create list of obstacles
    obstacles = enclosing_workspace_hull_polygon
    for polygon in polygon_list:
        obstacles = obstacles.difference(polygon)
    
    obstacles = list(obstacles)
    
    for i in range(len(obstacles)):
        obstacles[i] = obstacles[i].simplify(SIMPLIFICATION_CONSTANT, preserve_topology=True)
    
    
    diffeo_tree_array = []
    for i in range(len(obstacles)):
        coords = np.vstack((obstacles[i].exterior.coords.xy[0],obstacles[i].exterior.coords.xy[1])).transpose()
        diffeo_tree_array.append(diffeoTreeTriangulation(coords, diffeo_params))
    
    
    obstacles_decomposed_centers = []
    obstacles_decomposed_radii = []
    for tree in diffeo_tree_array:
        radius = tree[-1]['radius']
        center = tree[-1]['center'].squeeze()
        obstacles_decomposed_centers.append(center)
        obstacles_decomposed_radii.append(radius)
        
    return diffeo_tree_array, diffeo_params, obstacles_decomposed_centers, obstacles_decomposed_radii, obstacles, polygon_list, disjoint_sets, enclosing_workspace_hull_polygon

def create_obstacle_polygons(parameter_set_filtered):
    """
    Create obstacle polygons from filtered safe parameters.
    
    Args:
        parameter_set_filtered: Array of safe positions
        
    Returns:
        polygon_list: List of obstacle polygons
    """
    polygon_list = []
    
    # Build disjoint sets from the parameter set
    disjoint_sets = build_disjoint_sets(parameter_set_filtered, 1)
    
    # Create polygons from the disjoint sets
    for i, group in enumerate(disjoint_sets.subsets()):
        if len(group) > 2:
            points = [parameter_set_filtered[i] for i in group]
            points = np.array(points)
            
            # Create concave hull for larger groups
            if len(group) > 3:
                hull = concave_hull(points, concavity=1, length_threshold=0)
                poly = Polygon(hull)
            else:
                poly = Polygon(points) 
                
            polygon_list.append(poly)
    
    return polygon_list, disjoint_sets

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

def project_goal_to_polygon(goal, polygon):
    if polygon is not None:
        goal_point = Point(goal)
        if polygon.contains(goal_point):
            return goal  # If the goal is inside the polygon, no projection needed
        projected_point = polygon.boundary.interpolate(polygon.boundary.project(goal_point))
        return np.array([projected_point.x, projected_point.y])
    return goal

def compute_local_workspace_polygon(robot, robot_pos_transformed, obstacles_pos, obstacles_radii, screen_width, screen_height):

    # Gather the positions of the robot and obstacles
    
    points = [robot_pos_transformed] + obstacles_pos
    radii = [robot.radius] + obstacles_radii

    cells = pyvoro.compute_2d_voronoi(points, [[0, screen_width], [0, screen_height]], 2.0, radii=radii)

    return Polygon(cells[0]['vertices'])

def draw_environment(screen, surf, polygon_list, local_workspace_polygon, enclosing_workspace_hull_polygon, obstacles, next_parameters, BUFFER_SIZE):
    """
    Draw the environment on the screen.
    
    Args:
        screen: Pygame screen
        surf: Surface with environment data
        polygon_list: List of obstacle polygons
        next_parameters: Next sampling point
        path: Path for robot to follow
        BUFFER_SIZE: Scaling factor for display
    """
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    LIGHT_BLUE = (173, 216, 230)
    BLUE = (0, 0, 255)
    PINK = (255, 105, 180)
    
    # Clear the screen
    screen.fill(WHITE)
    
    screen.blit(surf, (0, 0))
    
    # Draw obstacle polygons
    for polygon in polygon_list:
        x, y = polygon.exterior.xy
        pygame.draw.polygon(screen, LIGHT_BLUE, np.array([x, y]).T * BUFFER_SIZE)
        
    pygame.draw.polygon(screen, GREEN, np.array(local_workspace_polygon.intersection(enclosing_workspace_hull_polygon).exterior.coords) * BUFFER_SIZE)
        
    for obstacle in obstacles:
        x, y = obstacle.exterior.xy
        pygame.draw.polygon(screen, BLACK, np.array([x, y]).T * BUFFER_SIZE)
        
    # Draw next parameters
    pygame.draw.circle(screen, PINK, 
                      (int(next_parameters[0])*BUFFER_SIZE, 
                       int(next_parameters[1])*BUFFER_SIZE), 
                      2*BUFFER_SIZE)
    

def modify_next_parameters(next_parameters, robot, polygon_list):
    buffered_polygon_list = []
    
    for polygon in polygon_list:
        buffered_polygon = polygon.buffer(-CONFIG["environment"]["SIMPLIFICATION_CONSTANT"])
        buffered_polygon_list.append(buffered_polygon)
        
    # make multipolygon
    buffered_polygon = sp.ops.unary_union(buffered_polygon_list)
    
    # project next parameters to the polygon
    
    if not buffered_polygon.contains(Point(next_parameters)):
        next_parameters = buffered_polygon.boundary.interpolate(buffered_polygon.boundary.project(Point(next_parameters)))
        next_parameters = np.array([next_parameters.x, next_parameters.y])
        
    return next_parameters


def draw_goal(screen, goal, BUFFER_SIZE):
    """
    Draw the goal point on the screen.
    
    Args:
        screen: Pygame screen
        goal: Goal position
        BUFFER_SIZE: Scaling factor for display
    """
    BLUE = (0, 0, 255)
    pygame.draw.circle(screen, BLUE, 
                      (int(goal[0])*BUFFER_SIZE, int(goal[1])*BUFFER_SIZE), 
                      2*BUFFER_SIZE)


import time

def main():
    # Try to load configuration from file
    load_config()
    
    connected = False
    # Load data
    FILENAME = CONFIG["environment"]["FILENAME"]
    Y = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
    Y_shape = Y.shape
    
    print ("Y shape: ", Y_shape)
    
    # Initialize Pygame
    pygame.init()
    
    # Display settings
    BUFFER_SIZE = CONFIG["display"]["BUFFER_SIZE"]
    screen_width = Y_shape[0] * BUFFER_SIZE
    screen_height = Y_shape[1] * BUFFER_SIZE
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Disk Robot Simulator")
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    LIGHT_BLUE = (173, 216, 230)
    BLUE = (0, 0, 255)
    PINK = (255, 105, 180)
    
    # Robot settings
    ROBOT_RADIUS = CONFIG["robot"]["ROBOT_RADIUS"]
    robot = Robot(230, 220, ROBOT_RADIUS, BLACK, screen_width, screen_height, BUFFER_SIZE=BUFFER_SIZE)
    
    # Goal settings
    goal = np.array([104, 96])
    
    # Setup the environment
    opt, parameter_set, parameter_set_filtered = setup_environment(Y, Y_shape, robot, goal)
    
    parameter_set_filtered_map = { tuple(param): i for i, param in enumerate(parameter_set_filtered) }
    
    
    next_parameters_orig = get_next_parameters(opt, goal, 1)
    
    diffeo_tree_array, diffeo_params, obstacles_decomposed_centers, obstacles_decomposed_radii, obstacles, polygon_list, disjoint_sets, enclosing_workspace_hull_polygon = setup_obstacles(parameter_set_filtered, robot)
    
    
    
    robot_pos_index = parameter_set_filtered_map.get((round(robot.pos[0]), round(robot.pos[1])))
    goal_pos_index = parameter_set_filtered_map.get((round(goal[0]), round(goal[1])))

    if goal_pos_index is not None and disjoint_sets.connected(robot_pos_index, goal_pos_index):
        next_parameters = goal
        connected = True
    else:
        next_parameters_orig = get_next_parameters(opt, goal, 1)
        next_parameters = modify_next_parameters(next_parameters_orig, robot, polygon_list)
    
    
    # Create surface for display
    max_Y = np.max(Y)
    min_Y = np.min(Y)
    Z = 255 * (Y - min_Y) / (max_Y - min_Y)
    Z = np.repeat(Z, BUFFER_SIZE, axis=1)
    Z = np.repeat(Z, BUFFER_SIZE, axis=0)
    
    # Create a surface with two colors based on the threshold
    surf = pygame.Surface((Z.shape[0], Z.shape[1]))
    for x in range(Z.shape[1]):
        for y in range(Z.shape[0]):
            if Y[y // BUFFER_SIZE, x // BUFFER_SIZE] > CONFIG["environment"]["THRESHOLD"]:
                surf.set_at((y, x), (255, 255, 255))  
            else:
                surf.set_at((y, x), (100, 100, 100)) 
    
    # Game loop variables
    clock = pygame.time.Clock()
    running = True
    update_robot = False
    last_keys = pygame.key.get_pressed()
    
    
    
    if CONFIG["environment"]["MODE"] == "semnav":
        mapped_goal, _, _ = transform_point(next_parameters, diffeo_tree_array, diffeo_params)
        
    
    
    
    # Main game loop
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    goal = np.array(pygame.mouse.get_pos())/BUFFER_SIZE
                    connected = False
                    print ("Goal set to: ", goal)
        
        # Handle key inputs
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and not last_keys[pygame.K_SPACE]:
            update_robot = not update_robot
        if keys[pygame.K_TAB] and not last_keys[pygame.K_TAB]:
            robot.clear_trail()
            robot.trail_enable = not robot.trail_enable
        last_keys = keys

        if CONFIG["environment"]["MODE"] == "semnav":
            mapped_pos, mapped_pos_d, _ = transform_point(robot.pos, diffeo_tree_array, diffeo_params)
            mapped_pos = mapped_pos.squeeze()
            mapped_pos_d = mapped_pos_d.squeeze()
            
            local_workspace_polygon = compute_local_workspace_polygon(robot, mapped_pos, obstacles_decomposed_centers, obstacles_decomposed_radii, screen_width, screen_height)
            
            local_free_space_polygon = local_workspace_polygon.buffer(-robot.radius)
            
            projected_goal = project_goal_to_polygon(mapped_goal.squeeze(), local_free_space_polygon)
        
            # Draw environment
            draw_environment(screen, surf, polygon_list, local_workspace_polygon, enclosing_workspace_hull_polygon, obstacles, next_parameters, BUFFER_SIZE)
            
            
            for i, value in enumerate(opt.G):
                if value:
                    x, y = parameter_set[i]
                    pygame.draw.circle(screen, RED, (int(x)*BUFFER_SIZE, int(y)*BUFFER_SIZE), 2*BUFFER_SIZE)
            
            
            
            # Update robot if enabled
            if update_robot:
                robot_vel_model = projected_goal - mapped_pos
                inverse_jacobian = np.linalg.pinv(mapped_pos_d)
                
                
                # claculate u
                u = np.dot(inverse_jacobian, robot_vel_model)
                
                print("u: ", np.linalg.norm(u))
                # Update robot position
                robot.update(u)
                
                # If reached final waypoint, update the model and replan
                if np.linalg.norm(u) < 0.1 and not connected:
                    # Add new data point at robot's position
                    opt.add_new_data_point(robot.pos, Y[round(robot.pos[0]), round(robot.pos[1])])
                    
                    # Update optimization
                    opt.update_confidence_intervals(context=None)
                    # opt.compute_sets(full_sets=False, num_expanders=NUM_EXPANDERS)
                    NUM_EXPANDERS = CONFIG["optimization"]["NUM_EXPANDERS"]
                    opt.compute_sets(goal=goal, num_expanders=NUM_EXPANDERS)
                    
                    # Update parameter set filtered
                    parameter_set_filtered = []
                    for index, value in enumerate(opt.S):
                        if value:
                            parameter_set_filtered.append(parameter_set[index])
                    
                    diffeo_tree_array, diffeo_params, obstacles_decomposed_centers, obstacles_decomposed_radii, obstacles, polygon_list, disjoint_sets, enclosing_workspace_hull_polygon = setup_obstacles(parameter_set_filtered, robot)
                    
                    parameter_set_filtered_map = { tuple(param): i for i, param in enumerate(parameter_set_filtered) }
                    robot_pos_index = parameter_set_filtered_map.get((round(robot.pos[0]), round(robot.pos[1])))
                    goal_pos_index = parameter_set_filtered_map.get((round(goal[0]), round(goal[1])))

                    
                    if goal_pos_index is not None and disjoint_sets.connected(robot_pos_index, goal_pos_index):
                        next_parameters = modify_next_parameters(goal, robot, polygon_list)
                        if np.linalg.norm(next_parameters - goal) < 0.1:
                            connected = True
                    else:
                        next_parameters_orig = get_next_parameters(opt, goal, 1)
                        next_parameters = modify_next_parameters(next_parameters_orig, robot, polygon_list)
                    
                    mapped_goal, _, _ = transform_point(next_parameters, diffeo_tree_array, diffeo_params)
                    
                    
        
        # Draw robot and goal
        robot.draw(screen)
        draw_goal(screen, goal, BUFFER_SIZE)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    # Quit pygame
    pygame.quit()


if __name__ == "__main__":
    main()