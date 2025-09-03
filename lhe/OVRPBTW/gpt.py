import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Heuristic for CVRP that ranks edges based on heuristic criteria."""
    num_nodes = distance_matrix.shape[0]
    capacity = 1.0  # Assuming normalized demands where total capacity behaves like 1.0
    edge_scores = torch.zeros_like(distance_matrix, dtype=torch.float)

    # Calculate total edge calls
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                # Calculate the demand for the arc from i to j
                demand_ij = demands[j]
                
                # Check if adding this demand exceeds capacity
                if demands[i] + demand_ij <= capacity:
                    # Favor shorter trips to customers with a manageable demand
                    edge_scores[i, j] = (capacity - (demands[i] + demand_ij)**2) / (distance_matrix[i, j] + 1e-6)
                else:
                    # Penalize if adding demand exceeds capacity
                    edge_scores[i, j] = -float('inf')  # effectively undesirable

    # Normalize scores: promoting promising edges and minimizing non-promising edges
    max_score = edge_scores.max()
    edge_scores[edge_scores == -float('inf')] = -1e6  # Replace undesirable scores with a large negative number
    edge_scores = (edge_scores - max_score) / (max_score + 1e-6)  # Scale to (-1,0] range

    return edge_scores
