import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

# File paths
BASE_DIR = '/content/drive/MyDrive/sentiment-aware-recommendation'
PROCESSED_REVIEWS_PATH = os.path.join(BASE_DIR, 'processed_reviews.csv')
KEY_PHRASES_PATH = os.path.join(BASE_DIR, 'key_phrases.txt')
BERT_MODEL_PATH = os.path.join(BASE_DIR, 'bert_sentiment_model')
MF_MODEL_PATH = os.path.join(BASE_DIR, 'mf_model.pth')
SENTIMENT_CACHE_PATH = os.path.join(BASE_DIR, 'sentiment_cache.csv')

# Initialize NLTK resources
def initialize_nltk():
    """Download required NLTK data if not already present."""
    nltk_data_path = '/root/nltk_data'
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource.startswith('punkt') else resource)
            print(f"NLTK resource {resource} found.")
        except LookupError:
            print(f"Downloading NLTK resource {resource}...")
            try:
                nltk.download(resource, download_dir=nltk_data_path, quiet=True)
                print(f"NLTK resource {resource} downloaded.")
            except Exception as e:
                print(f"Error downloading NLTK resource {resource}: {e}")
                raise Exception(f"Failed to download NLTK resource {resource}")

initialize_nltk()

# Text processing utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean text by removing URLs, special characters, and extra spaces."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\S+|[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text):
    """Tokenize and lemmatize text, removing stopwords."""
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

# Key phrase processing
def load_key_phrases():
    """Load key phrases from key_phrases.txt."""
    key_phrases = {'Positive': [], 'Negative': []}
    current_section = None
    
    try:
        with open(KEY_PHRASES_PATH, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Positive Phrases:'):
                current_section = 'Positive'
            elif line.startswith('Negative Phrases:'):
                current_section = 'Negative'
            elif current_section:
                try:
                    phrase, score = line.split(':')
                    key_phrases[current_section].append((phrase.strip(), float(score)))
                except ValueError:
                    print(f"Skipping malformed line: {line}")
        
        return key_phrases
    except FileNotFoundError:
        raise FileNotFoundError(f"Key phrases file not found at {KEY_PHRASES_PATH}")
    except Exception as e:
        raise Exception(f"Error loading key phrases: {e}")

def get_product_phrases(reviews, sentiment):
    """Extract primary and secondary key phrases from reviews."""
    phrases = load_key_phrases()[sentiment]
    review_text = ' '.join(clean_text(review) for review in reviews if isinstance(review, str))
    if not review_text:
        return {'primary': phrases[0][0] if phrases else f"{sentiment.lower()} review", 'secondary': []}
    
    review_tokens = tokenize_and_lemmatize(review_text)
    phrase_scores = []
    
    for phrase, pmi_score in phrases:
        phrase_tokens = tokenize_and_lemmatize(phrase)
        if not phrase_tokens:
            continue
        
        match_score = 0
        if all(token in review_tokens for token in phrase_tokens):
            positions = [i for i, t in enumerate(review_tokens) if t in phrase_tokens]
            if positions:
                proximity = max(positions) - min(positions) + 1
                match_score = len(phrase_tokens) / (proximity + 1)
        
        vectorizer = TfidfVectorizer()
        texts = [review_text, phrase]
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except Exception:
            similarity = 0.0
        
        final_score = (0.5 * match_score + 0.5 * similarity) * pmi_score
        phrase_scores.append((phrase, final_score))
    
    phrase_scores.sort(key=lambda x: x[1], reverse=True)
    result = {'primary': None, 'secondary': []}
    
    if phrase_scores:
        result['primary'] = phrase_scores[0][0]
        vectorizer = TfidfVectorizer()
        try:
            primary_vec = vectorizer.fit_transform([result['primary']])
            for phrase, score in phrase_scores[1:]:
                if score > 0 and len(result['secondary']) < 2:
                    secondary_vec = vectorizer.transform([phrase])
                    if cosine_similarity(primary_vec, secondary_vec)[0][0] < 0.7:
                        result['secondary'].append(phrase)
        except Exception:
            pass
    
    if not result['primary']:
        result['primary'] = phrases[0][0] if phrases else f"{sentiment.lower()} review"
    
    return result

# Matrix factorization model
class MatrixFactorization(nn.Module):
    """Matrix factorization model for collaborative filtering."""
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
    
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

def load_data():
    """Load and preprocess review data."""
    try:
        df = pd.read_csv(PROCESSED_REVIEWS_PATH)
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        df['user_id'] = user_encoder.fit_transform(df['reviewerID'])
        df['item_id'] = item_encoder.fit_transform(df['asin'])
        return df, user_encoder, item_encoder
    except FileNotFoundError:
        raise FileNotFoundError(f"Reviews file not found at {PROCESSED_REVIEWS_PATH}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def compute_sentiments(df):
    """Compute sentiment for reviews and cache results."""
    if os.path.exists(SENTIMENT_CACHE_PATH):
        try:
            sentiment_df = pd.read_csv(SENTIMENT_CACHE_PATH)
            if 'predicted_sentiment' in sentiment_df and 'sentiment_score' in sentiment_df:
                df['predicted_sentiment'] = sentiment_df['predicted_sentiment']
                df['sentiment_score'] = sentiment_df['sentiment_score']
                return df
        except Exception as e:
            print(f"Error loading sentiment cache: {e}")
    
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    def predict_sentiment(text):
        if not isinstance(text, str) or not text.strip():
            return 'Neutral'
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return label_map[probs.argmax(dim=-1).item()]
    
    df['predicted_sentiment'] = df['reviewText'].apply(predict_sentiment)
    df['sentiment_score'] = df['predicted_sentiment'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})
    
    # Cache sentiments
    sentiment_df = df[['predicted_sentiment', 'sentiment_score']]
    sentiment_df.to_csv(SENTIMENT_CACHE_PATH, index=False)
    
    return df

def train_mf_model(df, n_users, n_items):
    """Train and save matrix factorization model."""
    mf_model = MatrixFactorization(n_users, n_items)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mf_model.to(device)
    optimizer = torch.optim.Adam(mf_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    ratings = torch.FloatTensor(df['overall'].values).to(device)
    users = torch.LongTensor(df['user_id'].values).to(device)
    items = torch.LongTensor(df['item_id'].values).to(device)
    
    for epoch in range(10):
        mf_model.train()
        optimizer.zero_grad()
        predictions = mf_model(users, items)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    
    torch.save(mf_model.state_dict(), MF_MODEL_PATH)
    return mf_model

def get_recommendations(user_id, top_n=5):
    """
    Generate sentiment-aware recommendations for a user.
    
    Args:
        user_id (str): User ID (reviewerID).
        top_n (int): Number of recommendations to return.
    
    Returns:
        List of tuples: (asin, sentiment, phrases).
    """
    # Load data
    df, user_encoder, item_encoder = load_data()
    df = compute_sentiments(df)
    
    # Validate user_id
    try:
        encoded_user_id = user_encoder.transform([user_id])[0]
    except ValueError:
        raise ValueError(f"User ID {user_id} not found in reviews.")
    
    # Load or train MF model
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    
    mf_model = MatrixFactorization(n_users, n_items)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mf_model.to(device)
    
    if os.path.exists(MF_MODEL_PATH):
        try:
            mf_model.load_state_dict(torch.load(MF_MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"Error loading MF model: {e}. Retraining...")
            mf_model = train_mf_model(df, n_users, n_items)
    else:
        mf_model = train_mf_model(df, n_users, n_items)
    
    # Compute recommendations
    mf_model.eval()
    user_tensor = torch.LongTensor([encoded_user_id]).to(device)
    item_tensor = torch.LongTensor(range(n_items)).to(device)
    
    with torch.no_grad():
        scores = mf_model(user_tensor, item_tensor).cpu().numpy()
    
    # Adjust scores with sentiment
    item_sentiment = df.groupby('item_id')['sentiment_score'].mean().to_dict()
    for i in range(n_items):
        scores[i] = scores[i] * (1 + 0.5 * item_sentiment.get(i, 0))
    
    # Get top items
    top_items = np.argsort(scores)[::-1][:top_n]
    
    # Generate recommendations
    recommendations = []
    for item_id in top_items:
        asin = item_encoder.inverse_transform([item_id])[0]
        sentiment_score = item_sentiment.get(item_id, 0)
        if sentiment_score >= 0:
            sentiment = 'Positive' if sentiment_score > 0 else 'Neutral'
            sample_reviews = df[(df['asin'] == asin) & (df['predicted_sentiment'] == sentiment)]['reviewText'].dropna().tolist()
            phrases = get_product_phrases(sample_reviews, sentiment)
            recommendations.append((asin, sentiment, phrases))
    
    return recommendations

def main():
    """Test the recommendation system."""
    try:
        df, _, _ = load_data()
        user_id = df['reviewerID'].iloc[2090]  # Test with a valid user_id
        recommendations = get_recommendations(user_id, top_n=5)
        for asin, sentiment, phrases in recommendations:
            print(f"ASIN: {asin}, Sentiment: {sentiment}, Phrases: {phrases}")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()