import pandas as pd

def save_recommendations(recommendations, meta_data, output_path):
    output_rows = []
    for user_id, items in recommendations.items():
        for item_id in items:
            output_rows.append({
                'user_id': user_id,
                'pratilipi_id': item_id
            })
    
    output_df = pd.DataFrame(output_rows)
    output_with_meta = output_df.merge(meta_data[['pratilipi_id', 'category_name']], on='pratilipi_id', how='left')
    output_df.to_csv(output_path, index=False)
    return output_with_meta