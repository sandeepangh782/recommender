import pandas as pd

def load_data(user_data_path, meta_data_path):
    user_data = pd.read_csv(user_data_path)
    meta_data = pd.read_csv(meta_data_path)
    
    user_data['updated_at'] = pd.to_datetime(user_data['updated_at'])
    meta_data['updated_at'] = pd.to_datetime(meta_data['updated_at'])
    meta_data['published_at'] = pd.to_datetime(meta_data['published_at'])
    
    return user_data, meta_data