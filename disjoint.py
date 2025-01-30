import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from scipy.spatial import KDTree
from tqdm import tqdm

def build_disjoint_sets(points, threshold):
    """
    Build disjoint sets from a list of points using the Manhattan distance metric and a threshold.
    Utilizes KDTree for efficient neighbor searching.

    Parameters:
    - points (list of tuples): List of points, each tuple represents a point (e.g., [(x1, y1), (x2, y2), ...]).

    Returns:
    - disjoint_sets (dict): A dictionary where keys are component labels and values are lists of points in each set.
    """
    # Initialize DisjointSet
    ds = DisjointSet(range(len(points)))

    # Convert points to a NumPy array for KDTree
    points_array = np.array(points)

    # Build KDTree for efficient neighbor searching
    data_tree = KDTree(points_array)
    query_tree = KDTree(points_array)

    query_results = query_tree.query_ball_tree(data_tree, threshold)
    
    for i, neighbors in enumerate(tqdm(query_results, desc="Building disjoint sets")):
        for j in neighbors:
            if i < j:  # Only merge in one direction to avoid redundant checks
                ds.merge(i, j)

    

    return ds.subsets()

# Example usage
if __name__ == "__main__":
    points = [
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 2),
        (2, 3)
    ]
    threshold = 1

    result = build_disjoint_sets(points, threshold)

    print("Disjoint sets:")
    for label, group in result.items():
        print(f"Set {label}: {group}")
