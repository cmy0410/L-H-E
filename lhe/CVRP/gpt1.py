import torch
import torch
#15.640217781066895
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Calculate the maximum capacity
    vehicle_capacity = demands.sum()
    
    # Calculate savings: difference between direct distance and the sum of distances via the depot
    n = distance_matrix.shape[0]
    savings = torch.zeros_like(distance_matrix)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                savings[i, j] = distance_matrix[0, i] + distance_matrix[0, j] - distance_matrix[i, j]

    # Reward edges with high savings but penalize edges that would result in exceeding vehicle capacity
    heuristics_matrix = savings.clone()
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                # Demand penalty: the higher the combined demand, the less desirable the edge
                combined_demand = demands[i] + demands[j]
                if combined_demand > vehicle_capacity:
                    # Set a high penalty if the combined demand exceeds the vehicle capacity
                    heuristics_matrix[i, j] = -float('inf')
                else:
                    # Scale down the savings by the combined demand to node ratio
                    heuristics_matrix[i, j] -= combined_demand / vehicle_capacity * savings.max()

    return heuristics_matrix
