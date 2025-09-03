import torch
#9.053595542907715
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """A trivial implementation to improve upon."""
    return torch.zeros_like(distance_matrix)
