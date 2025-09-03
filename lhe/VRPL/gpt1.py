```python
import torch
#15.086478233337402
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Ensure demands do not exceed vehicle capacity, if demands[0] represents the depot, set it to 0
    demands[0] = 0
    
    # Calculate the inverse demand to prioritize nodes with lower demand
    inverse_demands = 1 / (demands + 1e-5)  # Adding a small constant to avoid division by zero
    
    # Initialize heuristic matrix with inverse demand values
    heuristic_matrix = torch.clone(inverse_demands).repeat(len(demands), 1)
    
    # Subtract distances from heuristic values to prioritize shorter routes
    heuristic_matrix -= distance_matrix
    
    # To discourage edges that exceed the vehicle capacity when combined with the current node's demand
    # First, we find the cumulative demand for each pair of nodes
    cumulative_demand_matrix = demands.repeat(len(demands), 1) + demands.repeat(len(demands), 1).t()
    
    # Assuming the capacity is normalized to 1, subtract large value for pairs exceeding capacity
    # This makes the heuristic value negative, indicating an undesirable edge
    heuristic_matrix[cumulative_demand_matrix > 1] = -torch.inf
    
    # The diagonal (visiting the same node) should not be considered
    torch.fill_diagonal(heuristic_matrix, -torch.inf)
    
    # Ensure that no nan or inf values are produced in the matrix
    heuristic_matrix[heuristic_matrix != heuristic_matrix] = -torch.inf
    heuristic_matrix[heuristic_matrix == float('inf')] = -torch.inf
    heuristic_matrix[heuristic_matrix == float('-inf')] = -1e5

    return heuristic_matrix
```
