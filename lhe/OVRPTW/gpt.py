import torch
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Enhanced heuristics for CVRP that incorporates demand-weighted distances and adaptive scoring."""
    n = distance_matrix.shape[0]
    vehicle_capacity = 1.0  # Normalized vehicle capacity
    max_demand = demands.max().item()

    # Initialize the heuristics matrix with negative values to indicate undesirable edges
    heurists = -torch.ones_like(distance_matrix)

    # Calculate demand-weighted scores and enforce conditions
    for i in range(1, n):
        if demands[i] <= vehicle_capacity:  # Valid customer check
            heurists[0, i] = 1 / (distance_matrix[0, i].item() * (demands[i] / max_demand + 1e-6))
            heurists[i, 0] = 1 / (distance_matrix[i, 0].item() * (demands[i] / max_demand + 1e-6))
            
            for j in range(1, n):
                if demands[i] + demands[j] <= vehicle_capacity and i != j:
                    # Encourage efficient routing between valid customers
                    heurists[i, j] = 1 / (distance_matrix[i, j].item() * (demands[i] + demands[j]) / (2 * max_demand + 1e-6))

    # Normalize the heuristics to map between 0 and 1 while avoiding NaN or Inf
    min_val = heurists.min()
    max_val = heurists.max()
    heurists = (heurists - min_val) / (max_val - min_val + 1e-6)

    return heurists
