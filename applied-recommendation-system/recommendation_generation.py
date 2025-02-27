from tqdm import tqdm

def generate_recommendations(model, train_matrix, user_to_idx, idx_to_user, idx_to_item):
    recommendations = {}
    user_ids = list(idx_to_user.keys())
    
    for user_idx in tqdm(user_ids, desc="Generating recommendations"):
        scores = model.recommend(
            user_idx,
            train_matrix[user_idx],
            N=5,
            filter_already_liked_items=True
        )
        item_indices, _ = scores
        user_id = idx_to_user[user_idx]
        recommended_items = [idx_to_item[item_idx] for item_idx in item_indices]
        recommendations[user_id] = recommended_items
    
    return recommendations