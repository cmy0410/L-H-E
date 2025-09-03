```python
import torch
#14.60425853729248
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Number of nodes (including depot)
    n = distance_matrix.size(0)
    
    # Vehicle capacity (assuming all demands are normalized by the same capacity)
    capacity = demands.sum()

    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Preventing the depot to be revisited (self-loops are not allowed)
    heuristic_matrix[torch.arange(n), torch.arange(n)] = -torch.inf

    # Set heuristic values for edges based on distance and demand
    # Promising edges are those that move towards less loaded vehicles
    # Penalize edges that would cause the vehicle to exceed its capacity
    for i in range(n):
        for j in range(n):
            if i != j:
                # The cost of servicing the node
                demand_cost = demands[j] / capacity
                
                # Edge penalty if capacity is exceeded
                if demands[i] + demands[j] > capacity:
                    heuristic_matrix[i, j] = -torch.inf
                else:
                    # The edge heuristic is negative of the distance
                    # We subtract the demand cost, prioritizing less loaded vehicles
                    heuristic_matrix[i, j] = -(distance_matrix[i, j] + demand_cost)

    # Ensure no NaNs or Infs are produced
    heuristic_matrix = torch.nan_to_num(heuristic_matrix, nan=-torch.inf, posinf=-torch.inf, neginf=-torch.inf)

    return heuristic_matrix
```
