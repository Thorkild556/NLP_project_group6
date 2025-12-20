from sklearn.neighbors import NearestNeighbors
import numpy as np


def nn_greedy_clustering(tfidf_matrix, min_size=2, max_size=4,
                          threshold=0.7, n_neighbors=20):
    """
    Use sklearn NN for fast neighbor finding, then greedy cluster formation
    """
    n = tfidf_matrix.shape[0]

    # Fit k-NN model (cosine similarity)
    nbrs = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n),  # +1 for self
                            metric='cosine',
                            algorithm='auto')
    nbrs.fit(tfidf_matrix)

    # Get neighbors for all samples at once
    distances, indices = nbrs.kneighbors(tfidf_matrix)
    similarities = 1 - distances  # Convert distance to similarity

    # Greedy clustering using precomputed neighbors
    assigned = np.zeros(n, dtype=bool)
    clusters = []

    # Process in order of maximum similarity (samples with strong matches first)
    max_sims = similarities[:, 1].copy()  # Skip self (index 0)
    order = np.argsort(max_sims)[::-1]

    for idx in order:
        if assigned[idx]:
            continue

        cluster = [int(idx)]  # Convert to Python int immediately

        # Add neighbors that are unassigned and above threshold
        # Start from index 1 to skip self (index 0 is always the sample itself)
        for neighbor_idx, sim in zip(indices[idx][1:], similarities[idx][1:]):
            neighbor_idx = int(neighbor_idx)  # Convert to Python int

            # if the post's similarity is above 95% we are discarding them
            # since they might be from the same ones but across subreddits
            if neighbor_idx != idx and not assigned[neighbor_idx] and sim >= threshold and sim < 0.95:
                cluster.append(neighbor_idx)
                if len(cluster) >= max_size:
                    break

        # Only add cluster if it meets minimum size
        if len(cluster) >= min_size:
            clusters.append(cluster)
            assigned[cluster] = True
        # If cluster too small, leave sample unassigned for now

    return clusters, assigned
