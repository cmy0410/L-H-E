import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Heuristic calculation for CVRP with positive and negative scoring for edges."""
    vehicle_capacity = 1  # Assume total vehicle capacity is normalized to 1 for the factors
    
    # Normalize distances to sulphate edge strictness
    distance_scaling = 1 / (distance_matrix + 1e-6)  # Avoiding division by zero
    
    # Create potential resource gain associated with visiting a node
    demand_penalty = demands.unsqueeze(1).expand_as(distance_matrix)  # Replicate demands for all rows
    resource_bias = vehicle_capacity - demand_penalty  # Positive bias where demands can be accepted
    
    # Combine distance gradient with resource dynamics
    # Subtract resource bias from distance values to consider the desirability of visiting a node
    scaler = distance_scaling * resource_bias
    
    # Apply a scaling factor based on desirability of logistics
    attractiveness = scaler * (scaler < 0) + -scaler * (scaler >= 0)
    
    return attractiveness
