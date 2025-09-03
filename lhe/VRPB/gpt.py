import torch
import numpy as np
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Enhanced heuristic for CVRP focusing on demand clustering and proximity metrics."""
    
    # Number of nodes
    n = distance_matrix.shape[0]
    
    # Calculate total demand and estimated vehicle capacity
    total_demand = demands.sum()
    estimated_vehicles = 3  # Adjust this based on actual scenario
    vehicle_capacity = total_demand / estimated_vehicles

    # Initialize heuristic value matrix
    heuristic_values = torch.zeros_like(distance_matrix)

    # Normalize distance matrix
    normalized_distance = distance_matrix / (distance_matrix.sum(dim=1, keepdim=True) + 1e-10)

    # Compute demand ratios
    demand_ratios = demands.unsqueeze(1) / vehicle_capacity  # Shape: (n, 1)

    # Evaluate heuristic values
    for i in range(n):
        for j in range(n):
            if i != j:
                total_demand = demands[i] + demands[j]
                if total_demand <= vehicle_capacity:
                    # Promising route due to feasible demand
                    heuristic_values[i, j] = (1 / normalized_distance[i, j]) * (vehicle_capacity - total_demand)
                else:
                    # Penalize routes that exceed vehicle capacity
                    heuristic_values[i, j] = - ((total_demand - vehicle_capacity) / normalized_distance[i, j])
    
    # Apply clustering penalty for nodes that cause a breach in vehicle capacity
    for i in range(1, n):
        potential_customers = (demands + demands[i] <= vehicle_capacity).nonzero(as_tuple=True)[0]
        if len(potential_customers) > 0:
            cluster_sum = demands[potential_customers].sum()
            if cluster_sum + demands[i] > vehicle_capacity:
                heuristic_values[i, potential_customers] -= 0.5  # Applying penalty for clustering violation

    # Set diagonal to negative values (unpromising for self-loops)
    heuristic_values.fill_diagonal_(-1.0)

    # Clamping values to manage extreme cases
    heuristic_values = torch.clamp(heuristic_values, min=-1.0, max=1.0)

    return heuristic_values
