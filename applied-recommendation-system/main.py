from data_loading import load_data
from data_preprocessing import preprocess_data
from model_training import train_model
from recommendation_generation import generate_recommendations
from evaluation import evaluate_model
from utils import save_recommendations
from visualization import visualize_data
from config import USER_DATA_PATH, META_DATA_PATH, OUTPUT_PATH

def get_user_recommendations(model, user_id, model_data, meta_data, top_n=10):
    """
    Get recommendations for a single user
    
    Args:
        model: Trained recommendation model
        user_id: User ID to get recommendations for
        model_data: Dictionary containing model data
        meta_data: Dictionary containing metadata
        top_n: Number of recommendations to return
    
    Returns:
        List of recommended items with their metadata
    """
    if user_id not in model_data['user_to_idx']:
        raise ValueError(f"User ID {user_id} not found in training data")
        
    user_idx = model_data['user_to_idx'][user_id]
    user_recommendations = generate_recommendations(
        model,
        model_data['train_matrix'],
        {user_id: user_idx},  # Only for specific user
        model_data['idx_to_user'],
        model_data['idx_to_item'],
        top_n=top_n
    )
    
    # Add metadata to recommendations
    recommendations_with_meta = []
    for item_id in user_recommendations[user_id]:
        item_metadata = meta_data.get(item_id, {})
        recommendations_with_meta.append({
            'item_id': item_id,
            'metadata': item_metadata
        })
    
    return recommendations_with_meta

def main():
    user_data, meta_data = load_data(USER_DATA_PATH, META_DATA_PATH)

    visualize_data(user_data, meta_data)
    
    model_data = preprocess_data(user_data, meta_data)
    

    model = train_model(model_data['train_matrix'])
    
 
    recommendations = generate_recommendations(
        model,
        model_data['train_matrix'],
        model_data['user_to_idx'],
        model_data['idx_to_user'],
        model_data['idx_to_item']
    )
    
    
    # evaluation_result = evaluate_model(
    #     model,
    #     model_data['test_data'],
    #     recommendations,
    #     model_data['user_to_idx'],
    #     model_data['idx_to_item']
    # )
    
    # print("Evaluation Results:")
    # print(evaluation_result)
    
   
    output_with_meta = save_recommendations(recommendations, meta_data, OUTPUT_PATH)
    print(f"Recommendations saved to {OUTPUT_PATH}")

    #to determine the recommendations for a specific user
    try:
        example_user_id = list(model_data['user_to_idx'].keys())[0]  # use userId instead of the key
        user_recs = get_user_recommendations(model, example_user_id, model_data, meta_data)
        print(f"\nRecommendations for user {example_user_id}:")
        for rec in user_recs:
            print(f"Item: {rec['item_id']}")
            print(f"Metadata: {rec['metadata']}\n")
    except Exception as e:
        print(f"Error getting user recommendations: {str(e)}")


if __name__ == "__main__":
    main()