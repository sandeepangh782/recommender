import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_data(user_data, meta_data):
    # Distribution of read percentage
    plt.figure(figsize=(10, 6))
    sns.histplot(user_data['read_percent'], bins=20)
    plt.title('Distribution of Read Percentage')
    plt.savefig('read_percentage_distribution.png')
    plt.close()

    # Category distribution
    category_counts = meta_data['category_name'].value_counts()
    plt.figure(figsize=(12, 8))
    category_counts[:20].plot(kind='bar')
    plt.title('Top 20 Categories')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('category_distribution.png')
    plt.close()

    # User activity distribution
    user_activity = user_data['user_id'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_activity, bins=50, log_scale=True)
    plt.title('User Activity Distribution (log scale)')
    plt.xlabel('Number of Interactions per User')
    plt.ylabel('Count')
    plt.savefig('user_activity_distribution.png')
    plt.close()

    # Item popularity distribution
    item_popularity = user_data['pratilipi_id'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(item_popularity, bins=50, log_scale=True)
    plt.title('Item Popularity Distribution (log scale)')
    plt.xlabel('Number of Interactions per Pratilipi')
    plt.ylabel('Count')
    plt.savefig('item_popularity_distribution.png')
    plt.close()
