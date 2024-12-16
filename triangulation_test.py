import sys
import pygame
import tripy
import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import split

def triangle_area(tri):
    p = Polygon(tri)
    return p.area

def share_edge(tri1, tri2):
    edges1 = set()
    for i in range(3):
        e = tuple(sorted([tri1[i], tri1[(i+1)%3]]))
        edges1.add(e)
    for i in range(3):
        e = tuple(sorted([tri2[i], tri2[(i+1)%3]]))
        if e in edges1:
            return True
    return False

def build_adjacency(triangles):
    n = len(triangles)
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if share_edge(triangles[i], triangles[j]):
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency

def bfs_tree(adjacency, start):
    """Perform BFS on the adjacency graph from the start node, return depths and parents."""
    from collections import deque
    queue = deque([start])
    depths = [-1]*len(adjacency)
    parents = [-1]*len(adjacency)
    depths[start] = 0
    visited = set([start])

    while queue:
        cur = queue.popleft()
        for neigh in adjacency[cur]:
            if neigh not in visited:
                visited.add(neigh)
                depths[neigh] = depths[cur] + 1
                parents[neigh] = cur
                queue.append(neigh)
    return depths, parents

def depth_to_color(depth):
    """Map a depth to a color. 
    """
    # Map depth to shades of blue, from lighter to darker
    max_depth = 5  # Adjust this based on your expected max depth
    
    base_color = (173, 216, 230)  # Light blue
    dark_color = (0, 0, 255)  # Dark blue
    
    # Interpolate between base_color and dark_color based on depth
    
    alpha = depth / max_depth
    
    color = tuple(int((1 - alpha) * c1 + alpha * c2) for c1, c2 in zip(base_color, dark_color))
    
    return color

def get_triangle_centroid(tri):
    x = (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0
    y = (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0
    return (x, y)


def get_admissible_center(triangles, child_index, parent_index):
    """Get a point inside the parent triangle along a median of the child triangle."""
    child_tri = triangles[child_index]
    parent_tri = triangles[parent_index]
    parent_poly = Polygon(parent_tri)

    # Get the median of the child triangle
    child_centroid = get_triangle_centroid(child_tri)
    
    # Get the shared edge between the child and parent
    shared_edge = None
    
    for i in range(3):
        edge = (child_tri[i], child_tri[(i+1)%3])
        if edge[0] in parent_tri and edge[1] in parent_tri:
            shared_edge = edge
            break
        
    if shared_edge is None:
        raise ValueError("Child triangle and parent triangle do not share an edge")
    
    shared_edge_midpoint = np.array([(shared_edge[0][0] + shared_edge[1][0]) / 2,
                                        (shared_edge[0][1] + shared_edge[1][1]) / 2])
    search_direction = shared_edge_midpoint - np.asarray(child_centroid)
    
    # normalize search direction
    
    search_direction = search_direction / np.linalg.norm(search_direction)
    
    parent_tri_max_width = max([np.linalg.norm(np.array(v) - shared_edge_midpoint) for v in parent_tri])
    
    candidates = np.linspace(0, parent_tri_max_width, num=10)
    
    for i in candidates:
        candidate = shared_edge_midpoint + i * search_direction
        if parent_poly.contains(Point(candidate)):
            return candidate
        
    raise ValueError("No admissible center found")

def get_polygonal_collar(triangles, child_index, parent_index):
    admissible_center = get_admissible_center(triangles, child_index, parent_index)
    triangle = triangles[child_index]
    
    # place the admissible center point between the shared vertices
    
    shared_vertices = []
    
    for i in range(3):
        if triangle[i] in triangles[parent_index]:
            if triangle[i] not in shared_vertices:
                shared_vertices.append(triangle[i])
        if triangle[(i+1)%3] in triangles[parent_index]:
            if triangle[(i+1)%3] not in shared_vertices:
                shared_vertices.append(triangle[(i+1)%3])

    
    non_shared_vertices = np.array([v for v in triangle])
    
    shared_vertices = np.array(shared_vertices)
    
    # get direction from admissible center to shared vertices
    
    directions = [shared_vertices[0] - admissible_center, shared_vertices[1] - admissible_center]\
    
    # normalize directions
    
    directions = [d / np.linalg.norm(d) for d in directions]
    
    # get the non shared vertex
    
    for sv in shared_vertices:
        # delete the shared vertices from the non shared vertices
        non_shared_vertices = [v for v in non_shared_vertices if not np.array_equal(v, sv)]
        
        
    non_shared_vertex = non_shared_vertices[0]
    
    
    print("non_shared_vertex", non_shared_vertex)
    
    poly = Polygon([shared_vertices[0], admissible_center, shared_vertices[1], non_shared_vertex])
    
    BUFFER_SIZE = 30
    CLIP_SIZE = 1000
    
    poly = poly.buffer(BUFFER_SIZE, join_style='bevel')
    
    bounding_box = [shared_vertices[0] + CLIP_SIZE*directions[0], admissible_center, shared_vertices[1] + CLIP_SIZE*directions[1], non_shared_vertex + CLIP_SIZE*directions[0]+CLIP_SIZE*directions[1]]
    
    poly = poly.intersection(Polygon(bounding_box))
    
    return poly
    
    
    
    
    


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Triangulation Test")
    clock = pygame.time.Clock()

    # Complex polygon
    polygon = [
        (-150, 50), (-100, 150), (-50, 100), (0, 150), (50, 100), (100, 150),
        (150, 50), (100, 0), (150, -50), (100, -150), (50, -100), (0, -150),
        (-50, -100), (-100, -150), (-150, -50), (-100, 0)
    ]

    # Triangulate using tripy
    triangles = tripy.earclip(polygon)

    # Find largest area triangle as root
    areas = [triangle_area(tri) for tri in triangles]
    largest_area_index = max(range(len(triangles)), key=lambda i: areas[i])

    # Build adjacency
    adjacency = build_adjacency(triangles)

    # BFS to get depths and parents
    depths, parents = bfs_tree(adjacency, largest_area_index)

    # Offset triangles to be centered on screen
    offset_x, offset_y = 400, 300
    polygon = [(x + offset_x, y + offset_y) for (x, y) in polygon]
    triangles = [[(vx+offset_x, vy+offset_y) for (vx, vy) in tri] for tri in triangles]

    # Compute admissible centers for all non-root triangles
    # Root has no parent
    admissible_centers = {}
    for i in range(len(triangles)):
        if i != largest_area_index and parents[i] != -1:
            admissible_centers[i] = get_admissible_center(triangles, i, parents[i])
            
    # Get the leaf triangles has highest depth
    
    # iterate over all triangles by depth
    
    map_depth_to_triangles = {}
    
    for i, depth in enumerate(depths):
        if depth not in map_depth_to_triangles:
            map_depth_to_triangles[depth] = []
        map_depth_to_triangles[depth].append(i)
        
    
    
    # Get the polygonal collar for each leaf triangle
    
    collars = {}
    
    # for depth, triangles_at_depth in map_depth_to_triangles.items():
    #     print("depth", depth)
    #     if depth > 0: 
    #         for leaf in triangles_at_depth:
    #             parent = parents[leaf]
    #             print(leaf, parent)
    #             collars[leaf] = get_polygonal_collar(triangles, leaf, parent)
    
    parent = parents[8]
    collars[8] = get_polygonal_collar(triangles, 8, parent)
    
    
    

    running = True
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # Draw original polygon outline
        pygame.draw.polygon(screen, (0, 0, 0), polygon, 2)
        
        # Draw the polygonal collars
        
        for i, collar in collars.items():
            pygame.draw.polygon(screen, (0, 100, 0), [(int(x), int(y)) for x, y in collar.exterior.coords])

        # Draw triangles colored by depth
        for i, tri in enumerate(triangles):
            c = depth_to_color(depths[i])
            pygame.draw.polygon(screen, c, tri, 0)
            pygame.draw.polygon(screen, (0,0,0), tri, 3)
            # label the triangle with its index
            font = pygame.font.SysFont("Helvetica", 24)
            text = font.render(str(i), True, (0, 0, 0))
            text_rect = text.get_rect(center=get_triangle_centroid(tri))
            screen.blit(text, text_rect)
            

        # # Draw admissible centers as small circles
        # for i, center_pt in admissible_centers.items():
        #     pygame.draw.circle(screen, (255,0,0), (int(center_pt[0]), int(center_pt[1])), 5)
            
        for i, collar in collars.items():
            pygame.draw.polygon(screen, (0, 255, 0), [(int(x), int(y)) for x, y in collar.exterior.coords], 3)
        

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
