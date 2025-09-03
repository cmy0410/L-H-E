import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Compute the euclidean distance matrix, if not already provided
    # Assuming distance_matrix is already provided and is valid.
    
    # Initialize the heuristic matrix with negative infinity to represent non-promising edges
    heuristic_matrix = torch.full_like(distance_matrix, float('-inf'))
    
    # Set the diagonal to a very negative value to exclude self-loops
    torch.fill_diagonal(heuristic_matrix, -float('inf'))
    
    # Avoid using the depot as an intermediate node
    heuristic_matrix[:, 0] = -float('inf')
    heuristic_matrix[0, :] = -float('inf')
    
    # Calculate the potential demand after serving a customer
    potential_demand = torchcumsum(torch.cat([torch.zeros(1), demands[:-1]]), dim=0) - demands
    
    # Calculate the heuristic scores
    # Positive values for promising edges, negative for undesirable ones
    # The heuristic score is based on the inverse distance and the capacity violation risk
    with torch.no_grad():
        # Inverse distance (penalty for longer distances)
        inv_distance = 1 / (distance_matrix + 1e-10)  # Adding a small constant to avoid division by zero
        # Capacity risk after serving a customer (penalty for overloading)
        capacity_risk = torch.clamp((potential_demand[:-1] - demands[1:]) / demands[1:], min=0)
        # Combine the inverse distance and the capacity risk
        heuristic_scores = inv_distance - capacity_risk
    
    # Clamp the heuristic scores to avoid nan or inf values
    heuristic_scores = torch.clamp(heuristic_scores, min=-float('inf'), max=float('inf'))
    
    # Copy the computed heuristic scores to the heuristic matrix where applicable
    heuristic_matrix[1:, 1:] = heuristic_scores
    
    return heuristic_matrix
