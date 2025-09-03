```python
import torch
#14.996719360351562
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    capacity = 1.0  # normalized vehicle capacity

    # Start with a matrix penalizing longer distances
    heuristic_matrix = -distance_matrix.clone()

    # Calculate the combined demands for all pairs of nodes
    combined_demands = demands.view(-1, 1) + demands.view(1, -1)
    
    # Mask for pairs exceeding the vehicle capacity
    capacity_violation_mask = combined_demands > capacity

    # Penalize capacity violations heavily
    heuristic_matrix[capacity_violation_mask] -= float('inf')

    # Normalize the heuristic matrix by the maximum distance to encourage feasible paths
    max_distance = torch.max(distance_matrix)
    heuristic_matrix /= max_distance

    # Encourage high capacity utilization and short depot routes
    # Increase heuristic value for nodes close to the depot and with high demand
    for i in range(1, n):  # Start from 1 to exclude the depot itself
        heuristic_matrix[i, 0] += demands[i] / capacity
        heuristic_matrix[0, i] += demands[i] / capacity

    # Ensure the depot has a strong incentive to return to itself
    heuristic_matrix[0, 0] = float('inf')
    
    # Replace NaNs and Infs resulting from operations
    heuristic_matrix[heuristic_matrix != heuristic_matrix] = -float('inf')
    heuristic_matrix[heuristic_matrix == float('inf')] = -float('inf')
    
    return heuristic_matrix
```
