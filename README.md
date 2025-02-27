# Pratilipi Recommendation System

This repository contains a hybrid recommendation system designed to predict which stories (pratilipis) users are likely to read in the future based on their historical reading behavior.

## 📖 Overview

The system combines collaborative filtering and content-based approaches to provide personalized recommendations. It analyzes user reading patterns, story metadata (categories, authors, reading time), and builds a model that can suggest relevant stories to users.

## 🚀 Features

- **Hybrid Approach**: Combines collaborative filtering (based on user behavior patterns) and content-based filtering (based on story attributes)
- **Time-based Evaluation**: Uses chronological split for training and testing (75%-25%)
- **Enhanced with Recency**: Considers publication dates to promote newer content when appropriate.

## 📂 Directory Structure

RECOMMENDER/
│── applied-recommendation-system/      # Core implementation of the recommendation system
│   ├── config.py                        # CSV paths
│   ├── data_loading.py                   # Handles data loading processes
│   ├── data_preprocessing.py              # Data cleaning and preprocessing
│   ├── evaluation.py                     # Evaluation metrics and performance analysis
│   ├── main.py                           # Main script to run the recommendation system
│   ├── model_training.py                  # Model training and optimization
│   ├── recommendation_generation.py       # Generating recommendations for users
│   ├── requirements.txt                   # List of dependencies
│   ├── utils.py                           # function to save the recommendatios to csv
│   ├── visualization.py                    # Visualization of results and data insights
│
│── experimental_approaches/               # Experimental recommendation techniques
│   ├── final_approach.ipynb               # Final tested approach for recommendation
│   ├── recommend_approach1.ipynb          # Alternative recommendation strategy 1
│   ├── recommendation_approach2.ipynb     # Alternative recommendation strategy 2
│   ├── recommendation-approach3.ipynb     # Alternative recommendation strategy 3
│
│── visualizations/                        # Stores visual representation outputs
│
│── LICENSE                                # License for project usage
│── README.md                              # Project documentation



## Data Description

The system uses two CSV files:

1. `User_interaction.csv`: Contains user interactions with pratilipis
   - user_id: Unique identifier of the user
   - pratilipi_id: Unique identifier of the pratilipi (story)
   - read_percent: Percentage of the pratilipi read by the user (0-100)
   - updated_at: Timestamp of the interaction

2. `Meta_data.csv`: Contains metadata about the pratilipis
   - author_id: Unique identifier of the author
   - pratilipi_id: Unique identifier of the pratilipi
   - category_name: Category of the pratilipi
   - reading_time: Reading time in seconds
   - updated_at: Timestamp of last update
   - published_at: Publication timestamp

## 📌 Setup Instructions

1. Place the dataset files in the same directory as the code
2. Install Dependencies:
```bash
pip install -r requirements.txt
```
3. Update the CSV paths in `config.py` to point to the correct dataset locations:
```python
# config.py
USER_INTERACTION_CSV_PATH = '/path/to/User_interaction.csv'
META_DATA_CSV_PATH = '/path/to/Meta_data.csv'
```
4. Run the main script:
```bash
python main.py
```


# Get 5 recommendations for specific user
```bash
#in main.py
example_user_id = list(model_data['user_to_idx'].keys())[0]  # use userId instead of the key
user_recs = get_user_recommendations(model, example_user_id, model_data, meta_data)
```

## ⚙️ Model Details

### Collaborative Filtering Component
- Uses Alternating Least Squares (ALS) algorithm
- Learns latent factors for users and items
- Captures implicit feedback through read percentages

### Content-Based Component
- Uses story metadata (categories)

### Hybrid Combination
- Weighted combination of both approaches (optimized through hyperparameter tuning)
- Enhanced with recency factors to balance between popular and new content

## Future Improvements

- Incorporate more advanced natural language processing on story content
- Use explicit feedback features such as pratilipi ratings to enhance the recommendation patterns

