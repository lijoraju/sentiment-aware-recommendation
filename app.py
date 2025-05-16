import streamlit as st
import pandas as pd
import torch
import os
from recommendation import get_recommendations
from explanations import generate_explanation

# File paths
BASE_DIR = '/content/drive/MyDrive/sentiment-aware-recommendation'
PROCESSED_REVIEWS_PATH = os.path.join(BASE_DIR, 'processed_reviews.csv')

def load_user_ids():
    """Load unique user IDs (reviewerID) from processed reviews."""
    try:
        df = pd.read_csv(PROCESSED_REVIEWS_PATH)
        if 'reviewerID' not in df.columns:
            st.error("Column 'reviewerID' not found in processed_reviews.csv")
            return []
        return sorted(df['reviewerID'].unique())
    except FileNotFoundError:
        st.error(f"Could not find {PROCESSED_REVIEWS_PATH}. Please check the file path.")
        return []
    except Exception as e:
        st.error(f"Error loading user IDs: {e}")
        return []

def main():
    """Main Streamlit app for sentiment-aware recommendations."""
    st.title("Sentiment-Aware Product Recommendations")
    st.write("Select a user ID to get personalized product recommendations with sentiment analysis and explanations.")

    # Load user IDs
    user_ids = load_user_ids()
    if not user_ids:
        st.warning("No user IDs found. Please ensure processed_reviews.csv is available.")
        return

    # User input
    user_id = st.selectbox("Select User ID", options=[""] + user_ids, index=0)
    
    if user_id:
        with st.spinner("Generating recommendations..."):
            try:
                # Get recommendations
                recommendations = get_recommendations(user_id, top_n=5)
                if not recommendations:
                    st.warning(f"No recommendations found for user {user_id}.")
                    return

                # Display recommendations
                st.subheader(f"Recommendations for User {user_id}")
                for idx, (asin, sentiment, phrases) in enumerate(recommendations, 1):
                    # Generate explanation
                    try:
                        explanation = generate_explanation(asin, sentiment, phrases)
                    except ValueError as ve:
                        explanation = f"Error generating explanation: {ve}"
                    except Exception as e:
                        explanation = f"Unexpected error generating explanation: {e}"

                    # Display recommendation details
                    with st.expander(f"Recommendation {idx}: Product {asin}"):
                        st.markdown(f"**Sentiment**: {sentiment}")
                        secondary_text = f", {', '.join(phrases['secondary'])}" if phrases['secondary'] else ""
                        st.markdown(f"**Key Phrases**: {phrases['primary']}{secondary_text}")
                        st.markdown(f"**Explanation**: {explanation}")
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except ValueError as ve:
                st.error(f"Error fetching recommendations: {ve}")
            except Exception as e:
                st.error(f"Unexpected error fetching recommendations: {e}")

if __name__ == "__main__":
    main()