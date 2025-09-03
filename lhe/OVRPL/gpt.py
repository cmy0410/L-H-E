import numpy as np
import numpy as np
from sklearn.cluster import KMeans

def heuristics_v2(distance_matrix: np.ndarray, demands: np.ndarray) -> np.ndarray:
    num_nodes = distance_matrix.shape[0]
    vehicle_capacity = demands[0]  # Vehicle capacity defined by depot's demand
    bias_matrix = -np.abs(distance_matrix)  # Initialize with negative values

    # Normalize demands (skipping depot)
    normalized_demands = demands[1:] / vehicle_capacity

    # Use KMeans for clustering based on normalized demands
    kmeans = KMeans(n_clusters=min(num_nodes - 1, len(normalized_demands) // 2))
    clusters = kmeans.fit_predict(np.column_stack((np.arange(1, num_nodes), normalized_demands)))

    # Calculate promising scores based on cluster distances and demands
    for cluster in set(clusters):
        cluster_indices = np.where(clusters == cluster)[0] + 1  # Adjust for demands

        # Intra-cluster checks
        for i in cluster_indices:
            for j in cluster_indices:
                if i != j:
                    total_demand = normalized_demands[i - 1] + normalized_demands[j - 1]
                    if total_demand <= 1:
                        distance_score = distance_matrix[i, j]
                        bias_matrix[i, j] = 1 / (distance_score + 1e-5)  # Favor closer distances
                        bias_matrix[j, i] = bias_matrix[i, j]  # Ensure symmetry

        # Check depot to customers
        for i in cluster_indices:
            if normalized_demands[i - 1] <= 1:
                depot_to_customer_distance = distance_matrix[0, i]
                bias_matrix[0, i] = 1 / (depot_to_customer_distance + 1e-5)
                bias_matrix[i, 0] = bias_matrix[0, i]

    # Normalize the heuristics for consistency
    mean = np.mean(bias_matrix)
    std = np.std(bias_matrix) + 1e-6
    bias_matrix = (bias_matrix - mean) / std

    # Ensure no NaN or Inf values
    bias_matrix = np.nan_to_num(bias_matrix)

    return bias_matrix
