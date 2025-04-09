from anytree import AnyNode, RenderTree
import numpy as np
from shapely.geometry import Point, LineString
import pygame

def is_segment_entirely_in_safe_region(line_segment, safe_region):
    """
    Check if a line segment stays completely within a safe region.
    
    Parameters:
    - line_segment: LineString object representing the segment
    - safe_region: Polygon object representing the safe region
    
    Returns:
    - True if segment is entirely within safe region, False otherwise
    """
    # Check if the line segment is completely contained in the safe region
    return safe_region.contains(line_segment)

def is_segment_safe(start_point, end_point, safe_regions):
    """
    Check if a line segment from start_point to end_point stays within any safe region.
    
    Parameters:
    - start_point: tuple (x, y) representing start of segment
    - end_point: tuple (x, y) representing end of segment
    - safe_regions: list of Polygon objects representing safe regions
    
    Returns:
    - True if segment is entirely within at least one safe region, False otherwise
    """
    # Create a LineString from the start and end points
    segment = LineString([start_point, end_point])
    
    # Check if the segment stays entirely within any safe region
    for region in safe_regions:
        if is_segment_entirely_in_safe_region(segment, region):
            return True
    
    # The segment exits all safe regions
    return False

def is_segment_safe_across_multiple_regions(start_point, end_point, safe_regions):
    """
    Check if every point on the line segment is within at least one safe region.
    """
    segment = LineString([start_point, end_point])

    # Create a union of all safe regions
    from shapely.ops import unary_union
    combined_safe_area = unary_union(safe_regions)
    
    # Check if the segment is completely contained in the combined safe area
    return combined_safe_area.contains(segment)

    
def find_nearest_node(root, point):
    nearest_node = root
    min_distance = np.linalg.norm(root.point - point)
    
    
    for pre, _, child in RenderTree(root):
        distance = np.linalg.norm(child.point - point)
        if distance < min_distance:
            min_distance = distance
            nearest_node = child
            
    return nearest_node

def steer(from_node, to_point, step_size):
    # points are np.array
    direction = to_point - from_node.point
    distance = np.linalg.norm(direction)
    if distance > step_size:
        direction = direction / distance * step_size
        new_point = from_node.point + direction
    else:
        new_point = to_point
    return new_point


    
def find_nodes_within_radius(tree, point, radius):
    """
    Find all nodes within a given radius from the point.
    
    Parameters:
    - tree: the root node of the tree
    - point: the point to check against
    - radius: the radius within which to find nodes
    
    Returns:
    - List of nodes within the radius
    """
    nodes_within_radius = []
    
    for pre, _, node in RenderTree(tree):
        if np.linalg.norm(node.point - point) <= radius:
            nodes_within_radius.append(node)
    
    return nodes_within_radius

def choose_best_parent(new_node, nearby_nodes):
    min_cost = float('inf')
    best_parent = None
    
    for node in nearby_nodes:
        # print (node)
        cost = node.cost + np.linalg.norm(node.point - new_node.point)
        if cost < min_cost:
            min_cost = cost
            best_parent = node
    
    if best_parent is not None:
        if new_node.parent is not best_parent:
            new_node.cost=min_cost
            new_node.parent = best_parent
        
def update_children_costs(node):
    for child in node.children:
        child.cost=node.cost + np.linalg.norm(child.point - node.point)
        update_children_costs(child)

def rewire_tree(new_node, nearby_nodes, safe_regions):
    for node in nearby_nodes:
        if node != new_node.parent and is_segment_safe(new_node.point, node.point, safe_regions):
            potential_cost = new_node.cost + np.linalg.norm(new_node.point - node.point)
            if potential_cost < node.cost:
                node.cost=potential_cost
                node.parent = new_node
                
                update_children_costs(node)
                

def extract_path(root, goal_node):
    path = []
    current_node = goal_node
    while current_node is not None:
        path.append(current_node.point)
        current_node = current_node.parent
    return path[::-1]  # Reverse the path to get it from start to goal

def rrt_star(start, goal, safe_set, safe_regions, max_iter=1000, step_size=10, radius=10, screen=None):
    start_node = AnyNode(point=start, cost=0) 
    print (start)
    goal_node = None
    for _ in range(max_iter):
        # Sample a random point in the safe set
        random_point = safe_set[np.random.choice(len(safe_set))]
        
        nearest_node = find_nearest_node(start_node, random_point)
        
        new_point = steer(nearest_node, random_point, step_size)
        
            
        if is_segment_safe_across_multiple_regions(nearest_node.point, new_point, safe_regions):
            
                
            # find nearby nodes within radius
            nearby_nodes = find_nodes_within_radius(start_node, new_point, radius)
            
            new_node = AnyNode(point=new_point, cost=0, parent=nearest_node)
            choose_best_parent(new_node, nearby_nodes)
            
            rewire_tree(new_node, nearby_nodes, safe_regions)
            if screen is not None:
                pygame.draw.line(screen, (0, 0, 255), tuple(new_node.parent.point), tuple(new_point))
            
            if np.linalg.norm(new_node.point - goal) < step_size and is_segment_safe(new_node.point, goal, safe_regions):
                goal_node = AnyNode(point=goal, parent=new_node)
                goal_node.cost = new_node.cost + np.linalg.norm(new_node.point - goal)
                break
                
                
    if goal_node is not None:
        path = extract_path(start_node, goal_node)
        return path
    else:
        return None
    
            
            
        
        
        
        
