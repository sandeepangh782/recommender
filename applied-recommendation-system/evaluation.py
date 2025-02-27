import numpy as np

def evaluate_model(model, test_data, recommendations, user_to_idx, idx_to_item):
    test_user_items = {}
    for user_id, group in test_data.groupby('user_id'):
        if user_id in user_to_idx:
            test_user_items[user_id] = set(group['pratilipi_id'])
    
    precisions = []
    recalls = []
    
    for user_id, actual_items in test_user_items.items():
        if user_id in recommendations:
            recommended_items = set(recommendations[user_id])
            true_positives = len(recommended_items.intersection(actual_items))
            precision = true_positives / len(recommended_items) if recommended_items else 0
            recall = true_positives / len(actual_items) if actual_items else 0
            precisions.append(precision)
            recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    result = {
        'precision': avg_precision,
        'recall': avg_recall,
        'user_coverage': len(precisions) / len(test_user_items)
    }
    
    return result