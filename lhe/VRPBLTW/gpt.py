import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Heuristic implementation for CVRP indicating promising edges based on distance and demand profiles."""
    n = distance_matrix.shape[0]
    
    # Calculate total demand normalization factor from vehicle capacity
    total_demand = demands.sum()
    vehicle_capacity = demands.max() * n  # Assuming capacity based on max demand
    normalized_demand = demands / vehicle_capacity
    
    # Initialize the promising indicator matrix
    performance_matrix = torch.zeros_like(distance_matrix)
    
    # Use a positive bias for shorter distances and a negative bias for over capacity edges
    for i in range(n):
        for j in range(n):
            if i != j:  # Skip self-loops
                # Calculate distance typically gets a positive score
                dist_score = 1 / (distance_matrix[i, j] + 1e-6)  # Prevent div by zero

                # Calculate the supply-demand ratio - penalizing edges that lead to combos over capacity
                if demands[i] + demands[j] <= vehicle_capacity:
                    demand_penalty = 0
                else:
                    demand_penalty = -1  # Penalty for exceeding capacity

                # Combine the two factors with a weighting mechanism
                performance_matrix[i, j] = (dist_score + demand_penalty) * (1 - normalized_demand[j])

    # Ensure no NaN or Inf values exist
    performance_matrix = torch.nan_to_num(performance_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    return performance_matrix
