import torch
import torch

def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    n = distance_matrix.shape[0]
    total_demand = demands.sum().item()
    vehicle_capacity = 1.0  # Assuming normalized demand is within [0, 1]

    # Initialize the heuristic matrix to zeros
    heuristic_matrix = torch.zeros_like(distance_matrix)

    # Dynamic clustering strategy based on demand
    cluster_threshold = vehicle_capacity / 2.0
    clusters = {}
    for i in range(1, n):
        key = (demands[i] // cluster_threshold).item()
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(i)

    # Historical performance indicator; starting with uniform values
    historical_performance = torch.ones_like(distance_matrix)

    # Evaluate edges based on clustering and distance
    for cluster_nodes in clusters.values():
        for i in cluster_nodes:
            for j in cluster_nodes:
                if i != j:
                    combined_demand = demands[i] + demands[j]
                    if combined_demand <= vehicle_capacity:
                        # Normalize distance value and add adaptive penalties
                        distance_penalty = 1.0 / (1.0 + distance_matrix[i, j])
                        heuristic_value = distance_penalty * (1.0 - (combined_demand / vehicle_capacity))
                        heuristic_matrix[i, j] = heuristic_value * historical_performance[i, j]
                    else:
                        heuristic_matrix[i, j] = -distance_matrix[i, j]  # Negative for infeasible edges

    # Incorporate a route feasibility check based on demands
    for i in range(1, n):
        for j in range(1, n):
            if i != j and distance_matrix[i, j] > 0 and demands[j] > vehicle_capacity:
                heuristic_matrix[i, j] = -torch.inf  # Mark as prohibited

    # Updating historical performance based on current heuristic matrix
    historical_performance += heuristic_matrix

    # Clip values to maintain the range and avoid inf values
    heuristic_matrix = torch.clamp(heuristic_matrix, min=-torch.inf, max=torch.inf)

    return heuristic_matrix
