import torch
def heuristics_v2(distance_matrix: torch.Tensor, demands: torch.Tensor) -> torch.Tensor:
    """Improved heuristics function for CVRP focusing on dynamic route savings and clustering considerations."""
    
    n = distance_matrix.shape[0]
    vehicle_capacity = torch.sum(demands[1:])  # Sum of customer demands for total capacity allocation
    promising_matrix = torch.zeros_like(distance_matrix)

    # Calculate distances from depot to customers
    depot_to_customers = distance_matrix[0, 1:]  # Extract distances from depot to customers
    nearest_indices = torch.argsort(depot_to_customers)[:n-1]  # Get indices of nearest customers
    cluster_size = min(5, n-1)  # Limit cluster size for consideration
    
    # Dynamic route savings - calculating potential savings from connecting two customers through the depot
    for i in range(1, n):
        for j in range(1, n):
            if i != j:  # Ignore self-loops
                edge_cost = distance_matrix[i, j]
                return_cost = distance_matrix[0, i] + distance_matrix[0, j]
                saving = return_cost - edge_cost  # Savings measure
                demand_sum = demands[i] + demands[j]

                # Apply clustering bonus for nearby nodes
                clustering_bonus = 0.0
                if i in nearest_indices[:cluster_size] or j in nearest_indices[:cluster_size]:
                    clustering_bonus = 0.5  # Favor connection among nearest nodes

                # Update promising matrix based on demand and savings
                if demand_sum > vehicle_capacity:  # If combined demands exceed vehicle capacity
                    promising_matrix[i, j] = -1 / (1 + edge_cost) + clustering_bonus - (demand_sum - vehicle_capacity) * 0.1
                else:
                    promising_matrix[i, j] = (saving / (1 + edge_cost)) + clustering_bonus

    # Normalize the promising matrix to avoid NaN or inf values and scale to [-1, 1]
    min_value, max_value = promising_matrix.min(), promising_matrix.max()
    if max_value > min_value:
        promising_matrix = (promising_matrix - min_value) / (max_value - min_value) * 2 - 1  # Map to [-1, 1]
    
    return promising_matrix
