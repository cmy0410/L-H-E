
import torch
#9.049043655395508
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    # Vehicle capacity is assumed to be 1 for normalization purposes
    n = distance_matrix.shape[0]
    # Initialize heuristics matrix with zeros
    heuristics_matrix = torch.zeros_like(distance_matrix)
    
    # Subtract the demand-based penalty from the distance to create incentives
    for i in range(n):
        for j in range(n):
            if i != j:  # We don't want to consider the depot to depot distance
                # Calculate a penalty proportional to the demand of the node j
                # The penalty is subtracted to make far nodes with high demand less desirable
                penalty = demands[j] * distance_matrix[i, j]
                # Apply penalty only if the demand of node j is greater than 0
                if demands[j] > 0:
                    heuristics_matrix[i, j] = distance_matrix[i, j] - penalty
    
    # Ensure that we never have NaN or inf values in the heuristic matrix
    heuristics_matrix = torch.clamp(heuristics_matrix, min=-float('inf'), max=float('inf'))
    
    # Ensure that the depot to depot edge is undesirable (negative value)
    heuristics_matrix[0, 0] = -float('inf')
    
    return heuristics_matrix

