# Sentiment-Aware Recommendation System

This project implements a sentiment-aware product recommendation system for Amazon Electronics Reviews. It combines collaborative filtering (matrix factorization), sentiment analysis (DistilBERT), key phrase extraction (statistical language model), and explanation generation (instruction-tuned GPT-2) to provide personalized product recommendations with user-friendly explanations. The system is deployed as an interactive Streamlit app, allowing users to input a `reviewerID` and receive top-5 product recommendations with sentiment-based explanations.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/lijoraju/sentiment-aware-recommendation.git
   cd sentiment-aware-recommendation
   ```

2. **Set Up Google Colab**:
   - Open Colab, mount Google Drive:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Copy the repository to `/content/drive/MyDrive/sentiment-aware-recommendation`.

3. **Install Dependencies**:
   ```bash
   !pip install streamlit==1.38.0 pandas==2.2.2 torch==2.4.0 transformers==4.44.2 numpy==1.26.4 scikit-learn==1.5.1 nltk==3.8.1 pyngrok
   ```

4. **Prepare Data and Models**:
   - Place `processed_reviews.csv`, `bert_sentiment_model/`, `gpt2_explanation_model/`, `mf_model.pth`, `key_phrases.txt`, `sentiment_cache.csv`, and `explanation_data.csv` in `/content/drive/MyDrive/sentiment-aware-recommendation`.
   - Note: These files are excluded from the repository due to size (see `.gitignore`).

5. **Run the Streamlit App**:
   ```python
   !streamlit run /content/drive/MyDrive/sentiment-aware-recommendation/app.py &>/dev/null&
   from pyngrok import ngrok
   !ngrok authtoken YOUR_NGROK_AUTH_TOKEN
   public_url = ngrok.connect(8501)
   print(f"Streamlit app running at: {public_url}")
   ```

## Dependencies

- Python 3.11
- `streamlit==1.38.0`
- `pandas==2.2.2`
- `torch==2.4.0`
- `transformers==4.44.2`
- `numpy==1.26.4`
- `scikit-learn==1.5.1`
- `nltk==3.8.1`
- `pyngrok`

## Usage

1. Open the Streamlit app via the ngrok URL.
2. Select a `reviewerID` from the dropdown (loaded from `processed_reviews.csv`).
3. View top-5 recommendations, including:
   - Product ASIN (e.g., `B000067O7T`).
   - Sentiment (Positive/Neutral).
   - Key phrases (e.g., “great camera, sturdy case”).
   - Explanation (e.g., “Product B000067O7T is recommended because...”).

## Example Output

```
Recommendation 4: Product B000067O7T
Sentiment: Positive
Key Phrases: great camera, sturdy case, case perfect
Explanation: Product B000067O7T is recommended because users found it to have a great camera, sturdy case, and perfect case, making it an excellent choice for photography enthusiasts.
```

## Evaluation

Run `evaluate.py` to compute:
- RMSE for matrix factorization recommendations.
- Precision and recall for sentiment classification.
```bash
python evaluate.py
```

## Contact

For questions or contributions, contact [lijoraju](https://github.com/lijoraju).