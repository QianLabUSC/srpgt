import pyvoro

# Define points
points = [[5.0, 7.0], [1.7, 3.2], [8.1, 6.3]]

# Define bounding box limits
limits = [[0.0, 10.0], [0.0, 10.0]]

# Set dispersion (block size)
dispersion = 2.0

# Compute Voronoi cells
cells = pyvoro.compute_2d_voronoi(
    points,
    limits,
    dispersion
)

# Extract vertices for the first point
point_index = 0
cell = cells[point_index]
vertices = cell['vertices']

print(f"Vertices of the Voronoi cell around point {points[point_index]}:")
for vertex in vertices:
    print(vertex)

