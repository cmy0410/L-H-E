import numpy as np
import numpy as np

def heuristics_v2(distance_matrix: np.ndarray) -> np.ndarray:
    num_nodes = distance_matrix.shape[0]
    heuristics_matrix = np.zeros_like(distance_matrix)

    # Apply a nonlinear transformation to the heuristics values
    heuristics_matrix = 1 / np.power(distance_matrix, 2)

    # Use a data-driven thresholding method
    threshold = np.mean(heuristics_matrix) + np.std(heuristics_matrix)
    heuristics_matrix[heuristics_matrix < threshold] = 0

    return heuristics_matrix
