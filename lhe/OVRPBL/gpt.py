import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Heuristic function for CVRP to suggest promising edges."""
    n = distance_matrix.size(0)
    vehicle_capacity = demands.sum().item()  # assuming total demand is treated as max vehicle capacity
    capacity_left = vehicle_capacity - demands.unsqueeze(1)  # calculate capacity left for each customer
    
    # Initialize the heuristics matrix
    heuristics_matrix = torch.zeros_like(distance_matrix, dtype=torch.float)

    # Score promising edges
    for i in range(n):
        for j in range(n):
            if i != j:
                if distance_matrix[i, j] < float('inf'):  # check for valid edge
                    if demands[j] <= capacity_left[i]:  # can we serve demand from i to j?
                        heuristics_matrix[i, j] = 1.0 / distance_matrix[i, j]  # inversely proportional to distance
                    else:
                        heuristics_matrix[i, j] = -1.0  # mark undesirable edges

    # Avoid NaN or Inf by clamping smaller values and normalizing the desirable edges
    heuristics_matrix[heuristics_matrix < 0] = float('-inf')  # for undesirable edges
    heuristics_matrix[heuristics_matrix == 0] = 1e-10  # for free edges to avoid division by zero
    heuristics_matrix_normalized = heuristics_matrix / (heuristics_matrix.max().item() + 1e-10)

    return heuristics_matrix_normalized
