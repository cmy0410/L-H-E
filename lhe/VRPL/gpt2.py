```python
import torch
#15.09012508392334
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # The number of nodes (including the depot)
    num_nodes = distance_matrix.shape[0]
    
    # The vehicle capacity (assuming all demands are normalized by the vehicle capacity)
    vehicle_capacity = torch.sum(demands)
    
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Loop over all pairs of nodes to calculate the heuristic for each edge
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Do not calculate for the depot to depot edge
                # Demand of the node j
                demand_j = demands[j]
                
                # Remaining capacity after visiting node j
                remaining_capacity = vehicle_capacity - demand_j
                
                # If there is not enough capacity to visit node j, make this edge undesirable
                if remaining_capacity < 0:
                    heuristics_matrix[i, j] = -torch.inf
                else:
                    # The inverse of distance as a positive heuristic value (the shorter the distance, the higher the heuristic)
                    # Subtract a small constant to ensure that no overflow occurs when inverting distances
                    small_constant = 1e-6
                    heuristics_matrix[i, j] = 1.0 / (distance_matrix[i, j] + small_constant)
                    
                    # Further adjustment: prefer nodes with smaller demand if all else is equal
                    heuristics_matrix[i, j] *= (1.0 / (demand_j + small_constant))
    
    # Replace any NaNs or Infs with very negative values to avoid numerical issues during optimization
    heuristics_matrix[heuristics_matrix != heuristics_matrix] = -torch.inf
    heuristics_matrix[heuristics_matrix == torch.inf] = -torch.inf
    
    return heuristics_matrix
```
