import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon

def create_workspace_with_holes(boundary, holes):
    """
    Create an obstacle that takes up the entire workspace except for holes.
    
    Parameters:
    boundary (Polygon): The boundary of the workspace
    holes (list): List of Polygon objects representing holes
    
    Returns:
    Polygon or MultiPolygon: The obstacle with holes
    """
    # Create the obstacle as the boundary minus the holes
    obstacle = boundary
    for hole in holes:
        obstacle = obstacle.difference(hole)
    
    return obstacle

def repulsive_potential_and_gradient(q, obstacle, eta=1.0, Q_star=0.2, epsilon=1e-6):
    """
    Calculate the repulsive potential and its gradient at a point q,
    for an obstacle that takes up the whole workspace and has holes.
    Assumes that the point is within one of the holes.
    
    Parameters:
    q (numpy.ndarray): The position vector (2D)
    obstacle (Polygon or MultiPolygon): The obstacle with holes
    eta (float): Scaling factor for the repulsive potential
    Q_star (float): Threshold distance
    epsilon (float): Small value for numerical gradient calculation
    
    Returns:
    tuple: (U_rep, grad_U_rep) - the repulsive potential and its gradient
    """
    # Convert q to a Shapely Point
    point = Point(q)
    
    # Calculate distance from point to obstacle
    D = obstacle.distance(point)
    
    if D == 0:
        return float('inf'), np.zeros(2)  # Point is inside the obstacle
    
    # Calculate gradient using finite differences
    grad_D = np.zeros(2)
    
    # Calculate gradient in x direction
    q_dx = np.array([q[0] + epsilon, q[1]])
    D_dx = obstacle.distance(Point(q_dx))
    grad_D[0] = (D_dx - D) / epsilon
    
    # Calculate gradient in y direction
    q_dy = np.array([q[0], q[1] + epsilon])
    D_dy = obstacle.distance(Point(q_dy))
    grad_D[1] = (D_dy - D) / epsilon
    
    # Normalize the gradient
    if np.linalg.norm(grad_D) > 1e-10:
        grad_D = grad_D / np.linalg.norm(grad_D)
    
    # Calculate the repulsive potential and its gradient using the formulas
    if D <= Q_star:
        # Repulsive potential formula: U_rep(q) = 1/2 * η * (1/D(q) - 1/Q*)²
        U_rep = 0.5 * eta * ((1.0 / D) - (1.0 / Q_star))**2
        
        # Gradient formula: ∇U_rep(q) = η * (1/Q* - 1/D(q)) * (1/D²(q)) * ∇D(q)
        factor = eta * ((1.0 / Q_star) - (1.0 / D)) * (1.0 / (D**2))
        grad_U_rep = factor * -grad_D  # Negative because gradient points away from obstacle
    else:
        # Zero potential and gradient beyond Q_star
        U_rep = 0.0
        grad_U_rep = np.zeros(2)
    
    return U_rep, grad_U_rep

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as mplPolygon
    
    # Define workspace boundary
    workspace_boundary = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    
    # Define holes
    hole1 = Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
    hole2 = Polygon([(6, 6), (8, 6), (8, 8), (6, 8)])

    
    # List of all holes
    holes = [hole1, hole2]
    
    # Create the obstacle (workspace minus holes)
    obstacle = create_workspace_with_holes(workspace_boundary, holes)
    
    # Test points (assume they're in holes)
    test_points = [
        np.array([3.4, 3]),    # Inside hole1
        np.array([7, 7]),    # Inside hole2
        np.array([5, 3]),    # Inside hole3
    ]
    
    # Calculate and print potential and gradient for each test point
    for i, q in enumerate(test_points):
        potential, gradient = repulsive_potential_and_gradient(q, obstacle, eta=1.0, Q_star=0.2)
        print(f"Point {i+1}: {q}")
        print(f"Distance to obstacle: {obstacle.distance(Point(q))}")
        print(f"Potential: {potential}")
        print(f"Gradient: {gradient}")
        print()
    
    # Visualize the workspace and potentials
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the obstacle
    if isinstance(obstacle, MultiPolygon):
        for geom in obstacle.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.5)
    else:
        x, y = obstacle.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5)
    
    # Plot the holes (these are not part of the obstacle)
    for hole in holes:
        x, y = hole.exterior.xy
        ax.plot(x, y, color='black')
    
    # Create a grid of points for the potential field
    x_grid = np.linspace(0, 10, 50)
    y_grid = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Calculate potential for each point in the grid
    Z = np.zeros_like(X)
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            q = np.array([X[i, j], Y[i, j]])
            point = Point(q)
            
            # Only calculate if point is in one of the holes (not in the obstacle)
            if not obstacle.contains(point):
                potential, gradient = repulsive_potential_and_gradient(q, obstacle, eta=1.0, Q_star=0.2)
                Z[i, j] = potential
                U[i, j] = gradient[0]
                V[i, j] = gradient[1]
    
    # Plot the potential field as a contour plot
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)
    fig.colorbar(contour, ax=ax, label='Repulsive Potential')
    
    # # Plot the gradient field as arrows
    # skip = 5  # Plot every 5th arrow for clarity
    # ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[::skip, ::skip], V[::skip, ::skip], 
    #           color='red', scale=20, width=0.002)
    
    # Plot the test points
    for q in test_points:
        potential, gradient = repulsive_potential_and_gradient(q, obstacle, eta=1.0, Q_star=0.2)
        ax.plot(q[0], q[1], 'ro', markersize=10)
        ax.quiver(q[0], q[1], gradient[0], gradient[1], color='red', scale=20, width=0.002)
    
    ax.set_title('Repulsive Potential Field in Workspace with Holes')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True)
    
    plt.savefig('repulsive_potential_field.png', dpi=300)
    plt.show()