import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from recommendation import load_data, compute_sentiments, MatrixFactorization
import torch

def evaluate_recommendations():
    # Load data
    df, user_encoder, item_encoder = load_data()
    df = compute_sentiments(df)

    # Split data (80% train, 20% test)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    # Train MF model
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    mf_model = MatrixFactorization(n_users, n_items)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mf_model.to(device)
    mf_model.load_state_dict(torch.load('/content/drive/MyDrive/sentiment-aware-recommendation/mf_model.pth', map_location=device))
    mf_model.eval()

    # Predict ratings
    test_users = torch.LongTensor(test_df['user_id'].values).to(device)
    test_items = torch.LongTensor(test_df['item_id'].values).to(device)
    with torch.no_grad():
        predictions = mf_model(test_users, test_items).cpu().numpy()
    true_ratings = test_df['overall'].values

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
    print(f"RMSE: {rmse:.4f}")

    # Evaluate sentiment accuracy
    true_sentiments = test_df['overall'].map(lambda x: 'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral')
    pred_sentiments = test_df['predicted_sentiment']
    precision = precision_score(true_sentiments, pred_sentiments, average='weighted', zero_division=0)
    recall = recall_score(true_sentiments, pred_sentiments, average='weighted', zero_division=0)
    print(f"Sentiment Precision: {precision:.4f}")
    print(f"Sentiment Recall: {recall:.4f}")

if __name__ == "__main__":
    evaluate_recommendations()