```python
import torch
#14.61925220489502
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    vehicle_capacity = demands[0]

    # Calculate inverse distance matrix to prioritize closer nodes
    inv_distance_matrix = 1 / (distance_matrix + 1e-6)

    # Normalize demands to vehicle capacity for balance
    normalized_demands = demands / vehicle_capacity

    # Create a demand penalty matrix based on normalized demands
    demand_penalty_matrix = normalized_demands.unsqueeze(0).expand_as(distance_matrix) * normalized_demands.unsqueeze(1).expand_as(distance_matrix)

    # Mask to heavily penalize edges where combined demand exceeds vehicle capacity
    capacity_excess_mask = (demand_penalty_matrix > 1).float()
    
    # Demand penalty factoring in the proximity; closer demands have less penalty
    demand_penalty = demand_penalty_matrix * inv_distance_matrix * capacity_excess_mask

    # Initialize heuristic matrix with inverse distances and subtract penalties
    heuristic_matrix = inv_distance_matrix - demand_penalty

    # Mask the depot edges, heavily penalize to prevent routing via depot
    heuristic_matrix[:, 0] = -1e9
    heuristic_matrix[0, :] = -1e9

    # Prevent routing to the same node by heavily penalizing the diagonal
    torch.diagonal(heuristic_matrix)[:] = -1e9

    # Apply capacity constraint: set heuristic values to negative for any capacity violation
    heuristic_matrix[capacity_excess_mask.bool()] = -1e9

    # Normalize the heuristic matrix so that the most promising routes have the highest positive values
    max_val = heuristic_matrix[heuristic_matrix > -1e9].max()
    heuristic_matrix -= (heuristic_matrix - max_val).min()

    return heuristic_matrix
```
