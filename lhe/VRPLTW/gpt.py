import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Heuristic implementation for Capacitated Vehicle Routing Problem (CVRP)."""
    # Ensure demands do not push vehicle over capacity
    vehicle_capacity = 1.0  # assuming normalized capacity
    n = distance_matrix.shape[0]
    
    # Compute a vehicle capacity usable matrix
    capacity_matrix = (demands.unsqueeze(1) + demands.unsqueeze(0)) <= vehicle_capacity
    
    # Importance of edges: if nodes demands exceed vehicle capacity return a large negative value
    penalty_value = -torch.inf
    heuristics_matrix = torch.where(capacity_matrix, 1.0 / (distance_matrix + 1e-9), penalty_value)
    
    # Normalize heuristics measures, skipping the depot (node 0)
    heuristics_matrix[0, :] = heuristics_matrix[:, 0] = 0.0  # depot has no heuristics
    
    # Introducing some negative influence based on distances alone to differentiate edges further
    distance_negative_scaling = -torch.exp((distance_matrix - distance_matrix.mean(axis=1, keepdim=True)) / (distance_matrix.std(axis=1, keepdim=True) + 1e-9))
    
    # Combine capacity usage heuristics and distance influence
    final_heuristics = heuristics_matrix + distance_negative_scaling
    
    return final_heuristics
