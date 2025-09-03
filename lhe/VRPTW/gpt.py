import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    vehicle_capacity = 1.0  # Assume normalized capacity
    n = distance_matrix.size(0)
    
    # Initialize the bias matrix
    bias_matrix = torch.zeros((n, n))

    # Calculate the valid edges based on demand and vehicle capacity
    for i in range(n):
        for j in range(1, n):  # Starting from 1 to skip the depot (0)
            if i != j:
                if demands[j] <= vehicle_capacity:
                    # Calculate distance and demand effect
                    distance_effect = 1 / (distance_matrix[i, j] + 1e-5)  # Avoid division by zero
                    demand_effect = 1 / (demands[j] + 1e-5)  # Avoid division by zero
                    bias_matrix[i, j] = distance_effect * demand_effect  # Combined score for edge
                else:
                    bias_matrix[i, j] = -1  # Penalize infeasible edges

    # Normalize the bias matrix for promising edges
    max_value = bias_matrix.max()
    min_value = bias_matrix.min()

    if max_value > min_value:  # Prevent division by zero
        bias_matrix = (bias_matrix - min_value) / (max_value - min_value)
    
    # Adjust negative values: retain negative scores for infeasible edges
    bias_matrix[bias_matrix < 0] = -1  # Set undesired edges to a constant penalty

    return bias_matrix
