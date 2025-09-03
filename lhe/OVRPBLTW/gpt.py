import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Improved heuristic for CVRP assessing edge inclusion potential based on distance and demand ratios."""
    # Mask to prevent invalid operations like division by zero
    non_zero_distances = distance_matrix.clone()
    non_zero_distances[distance_matrix == 0] = 1e-5

    # Calculate capacity utilization for each node considering the depot as node 0
    total_capacity = demands.sum()
    normalized_demands = demands / total_capacity

    # Calculate promisingness based on distance scaled by demand ratios
    promisingness_matrix = (normalized_demands.unsqueeze(1) * distance_matrix) / (1 + normalized_demands.unsqueeze(1))
    
    # Normalize promisingness values to maintain them in a manageable numeric range
    max_value = promisingness_matrix.max()
    min_value = promisingness_matrix.min()
    
    # Scale all promisingness values into range [0, 1] then shift to [-1, 1]
    promisingness_matrix = 2 * (promisingness_matrix - min_value) / (max_value - min_value) - 1 

    # Introducing a penalty for direct requests on depot (0 index)
    promisingness_matrix[0, :] = promisingness_matrix[:, 0] = -1

    return promisingness_matrix
