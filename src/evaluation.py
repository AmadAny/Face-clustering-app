import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_clustering(embeddings, cluster_labels, true_labels=None):
    metrics = {}
    
    valid_mask = cluster_labels != -1
    if np.sum(valid_mask) > 1 and len(np.unique(cluster_labels[valid_mask])) > 1:
        try:
            metrics['silhouette'] = silhouette_score(embeddings[valid_mask], cluster_labels[valid_mask])
        except Exception:
            metrics['silhouette'] = 'N/A'
    else:
        metrics['silhouette'] = 'N/A'
    
    if true_labels is not None and len(true_labels) == len(cluster_labels):
        metrics['ari'] = adjusted_rand_score(true_labels, cluster_labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, cluster_labels)
    
    return metrics
