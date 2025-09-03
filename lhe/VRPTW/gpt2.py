```python
import torch
#26.50217628479004
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Initialize the heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)

    # Calculate the inverse demand, higher demand will result in smaller values
    inv_demand = 1.0 / (demands + 1e-5)  # Adding a small constant to avoid division by zero

    # Subtract the inverse demand from the distance matrix to create the heuristics
    # The intuition here is that higher demand nodes are less desirable to connect directly
    # unless they are very close, hence the penalty in the form of inverse demand.
    heuristics_matrix -= torch.diag(inv_demand) + torch.diag(inv_demand)[:, None]

    # Nodes that are closer together should be favored, hence the negative distances
    # are used to indicate more promising edges.
    heuristics_matrix -= distance_matrix

    # Zero out the diagonal (self-loops are not allowed)
    torch.diagonal(heuristics_matrix).zero_()

    # Ensure no Inf or NaN values are produced
    heuristics_matrix = torch.nan_to_num(heuristics_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize the heuristics to avoid too large values
    heuristics_matrix /= heuristics_matrix.max()

    return heuristics_matrix
```
