```python
import torch
#10.293614387512207
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.size(0)
    customer_demands = demands[1:]
    
    heuristic_matrix = -distance_matrix.clone()
    torch.fill_(heuristic_matrix.diag(), -torch.inf)

    inverse_demands = 1 / (customer_demands + 1e-5)
    
    heuristic_matrix[1:, 1:] *= torch.diagflat(inverse_demands)
    
    # Overcapacity penalty factor
    overcapacity_penalty_factor = 100  # This factor can be tuned

    # Calculate the cumulative demand matrix
    cumulative_demand_matrix = demands.view(-1, 1) + demands.view(1, -1)
    
    # Apply a heavy penalty when the sum of demands between two nodes exceeds the vehicle capacity
    overcapacity_penalty = torch.where(cumulative_demand_matrix > 1, 
                                       overcapacity_penalty_factor * (cumulative_demand_matrix - 1), 
                                       torch.zeros_like(cumulative_demand_matrix))
    
    heuristic_matrix -= overcapacity_penalty

    # Penalize the depot-to-depot edge separately
    heuristic_matrix[0, 0] = -torch.inf

    # Penalize edges that could potentially create loops by setting the diagonal to -inf after adjustments
    torch.fill_(heuristic_matrix.diag(), -torch.inf)

    # Ensure no NaN or inf values
    heuristic_matrix[torch.isnan(heuristic_matrix) | torch.isinf(heuristic_matrix)] = -torch.inf

    return heuristic_matrix
```
