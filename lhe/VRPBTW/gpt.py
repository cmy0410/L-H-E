import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """A more advanced heuristic for the Capacitated Vehicle Routing Problem (CVRP)."""
    n = distance_matrix.shape[0]
    total_capacity = torch.sum(demands[1:])  # Exclude the depot's demand
    promising_scores = torch.zeros_like(distance_matrix)

    # Evaluate edges: positive score for valid edges and negative otherwise
    for i in range(n):
        for j in range(n):
            if i != j:
                demand_sum = demands[i] + demands[j]
                # If the total demand in the edge is within capacity, we give a positive score
                if i == 0 or j == 0:  # One node is depot
                    if demands[j] <= 1:  # Capacity normalized
                        promising_scores[i, j] = 1.0 / (1 + distance_matrix[i, j])
                else:
                    if demand_sum <= 1:  # Both customers are valid to be served by one vehicle
                        promising_scores[i, j] = 1.0 / (1 + distance_matrix[i, j])
                    else:  # This edge is undesirable due to capacity issues
                        promising_scores[i, j] = -1.0 / (1 + distance_matrix[i, j])

    # Clipping scores to avoid nan or inf
    promising_scores = torch.clamp(promising_scores, min=-10.0, max=10.0)
    return promising_scores
