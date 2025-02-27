import pandas as pd
from scipy.sparse import csr_matrix

def preprocess_data(user_data, meta_data):
    user_data['read_percent_norm'] = user_data['read_percent'] / 100.0
    max_date = user_data['updated_at'].max()
    user_data['days_since_interaction'] = (max_date - user_data['updated_at']).dt.days
    max_days = user_data['days_since_interaction'].max()
    user_data['recency_weight'] = 1 - (user_data['days_since_interaction'] / max_days)
    user_data['engagement_score'] = user_data['read_percent_norm'] * (0.7 + 0.3 * user_data['recency_weight'])
    
    enhanced_data = user_data.merge(meta_data, on='pratilipi_id', how='left')
    
    reading_time_quantiles = meta_data['reading_time'].quantile([0.33, 0.66]).values
    meta_data['length_category'] = pd.cut(
        meta_data['reading_time'], 
        bins=[0, reading_time_quantiles[0], reading_time_quantiles[1], float('inf')],
        labels=['short', 'medium', 'long']
    )
    
    meta_data['content_age_days'] = (max_date - meta_data['published_at']).dt.days
    
    user_ids = user_data['user_id'].unique()
    item_ids = meta_data['pratilipi_id'].unique()
    
    user_to_idx = {user: i for i, user in enumerate(user_ids)}
    item_to_idx = {item: i for i, item in enumerate(item_ids)}
    idx_to_user = {i: user for user, i in user_to_idx.items()}
    idx_to_item = {i: item for item, i in item_to_idx.items()}
    
    rows = [user_to_idx[user] for user in user_data['user_id']]
    cols = [item_to_idx.get(item, 0) for item in user_data['pratilipi_id']]
    data = user_data['read_percent'].values / 100.0
    
    user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    
    user_data_sorted = user_data.sort_values('updated_at')
    split_idx = int(len(user_data_sorted) * 0.75)
    train_data = user_data_sorted.iloc[:split_idx]
    test_data = user_data_sorted.iloc[split_idx:]
    
    train_rows = [user_to_idx[user] for user in train_data['user_id']]
    train_cols = [item_to_idx.get(item, 0) for item in train_data['pratilipi_id']]
    train_data_values = train_data['read_percent'].values / 100.0
    
    train_matrix = csr_matrix((train_data_values, (train_rows, train_cols)), shape=(len(user_ids), len(item_ids)))
    
    all_categories = meta_data['category_name'].unique()
    category_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    
    item_category_data = []
    for _, row in meta_data.iterrows():
        item_idx = item_to_idx.get(row['pratilipi_id'])
        if item_idx is not None:
            category_idx = category_to_idx.get(row['category_name'])
            if category_idx is not None:
                item_category_data.append((item_idx, category_idx, 1.0))
    
    item_cat_rows = [i for i, _, _ in item_category_data]
    item_cat_cols = [j for _, j, _ in item_category_data]
    item_cat_data = [v for _, _, v in item_category_data]
    
    item_category_matrix = csr_matrix((item_cat_data, (item_cat_rows, item_cat_cols)), shape=(len(item_ids), len(all_categories)))
    
    model_data = {
        'user_item_matrix': user_item_matrix,
        'train_matrix': train_matrix,
        'test_data': test_data,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item,
        'item_category_matrix': item_category_matrix,
        'category_to_idx': category_to_idx
    }
    
    return model_data