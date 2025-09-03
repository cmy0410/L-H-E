import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Advanced heuristics to evaluate edge promise in CVRP with adaptive learning and penalties."""
    
    vehicle_capacity = demands[0]  # Assuming the first node is the depot
    n_customers = demands.shape[0]

    # Calculate an inverted distance score: closer nodes are more promising
    distance_scores = 1 / (distance_matrix + 1e-5)  # Small epsilon to avoid division by zero

    # Assess capacity utilization and apply penalties if vehicle capacity is exceeded
    capacity_exceed_penalty = (demands.view(-1, 1) + demands.view(1, -1) > vehicle_capacity).float() * 1e3

    # Aggregate scores with a focus on maximizing distance efficiency and minimizing overlaps
    heuristics_scores = distance_scores - capacity_exceed_penalty

    # Normalize heuristic scores to keep them in bounds
    heuristics_scores[torch.isinf(heuristics_scores)] = 0
    heuristics_scores[torch.isnan(heuristics_scores)] = 0
    max_score = heuristics_scores.max() + 1e-5
    heuristics_scores = heuristics_scores / max_score

    # Implement adaptive learning: reward edges that contribute to greater demand fulfillment
    demand_fulfillment = demands.view(-1, 1) + demands.view(1, -1)  # Total demand of both nodes
    fulfilled_demand_scores = (vehicle_capacity - demand_fulfillment + 1e-6) * (demand_fulfillment <= vehicle_capacity)
    
    # Combine scores while penalizing overlaps
    overlap_penalty = torch.eye(n_customers).to(distance_matrix.device) * -1e3  # Heavy penalty on self-loops
    heuristics_scores += fulfilled_demand_scores - overlap_penalty

    # Introduce dynamic scaling based on historical performance (if available)
    # For demonstration, consider leveraging an arbitrary scaling factor: could be replaced with real historical data
    historical_scaling_factor = 1 + (1 - (demands[1:].sum() / (vehicle_capacity - demands[0] + 1e-5)))
    heuristics_scores *= historical_scaling_factor

    return heuristics_scores
