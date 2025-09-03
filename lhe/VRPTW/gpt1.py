```python
import torch
#26.436126708984375
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristic matrix with zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)
    
    # Calculate the inverse of demand (smaller demand, higher the value)
    inverse_demand = 1.0 / (demands + 1e-5)  # Adding a small value to avoid division by zero
    inverse_demand[0] = 0  # Set the depot's inverse demand to zero

    # Create a matrix where each element (i, j) is the sum of inverse demands of node i and j
    # This will give higher values for edges connecting nodes with smaller demands
    demand_sum_matrix = torch.mm(inverse_demand.unsqueeze(0), inverse_demand.unsqueeze(1))

    # The heuristic value for each edge (i, j) is inversely proportional to the distance,
    # and proportional to the sum of inverse demands (to encourage visiting nodes with smaller demand).
    # Also, we set the diagonal to negative infinity since it's not possible for a node to connect to itself in CVRP
    heuristic_matrix = -(distance_matrix / (demand_sum_matrix + 1e-5))
    torch.diagonal(heuristic_matrix).fill_(-torch.inf)

    # Ensure that we do not produce NaN or inf values
    heuristic_matrix[heuristic_matrix != heuristic_matrix] = -torch.inf
    heuristic_matrix[heuristic_matrix == torch.inf] = -torch.inf

    # Return the heuristic matrix
    return heuristic_matrix
```
