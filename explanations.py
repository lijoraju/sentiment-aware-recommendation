import os
import re
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# File paths
BASE_DIR = '/content/drive/MyDrive/sentiment-aware-recommendation'
MODEL_PATH = os.path.join(BASE_DIR, 'gpt2_explanation_model')
DATA_PATH = os.path.join(BASE_DIR, 'explanation_data.csv')
TRAINING_OUTPUT_DIR = os.path.join(BASE_DIR, 'gpt2_training')

# Global model and tokenizer for reuse
_model = None
_tokenizer = None

class ExplanationDataset(Dataset):
    """Dataset for fine-tuning GPT-2 with prompt-explanation pairs."""
    def __init__(self, data: pd.DataFrame, tokenizer: GPT2Tokenizer, max_length: int = 512):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        
        for idx, row in data.iterrows():
            prompt = row.get("prompt")
            explanation = row.get("explanation")
            if not (isinstance(prompt, str) and isinstance(explanation, str) and prompt.strip() and explanation.strip()):
                print(f"Skipping invalid row {idx}: prompt={prompt}, explanation={explanation}")
                continue
            text = f"{prompt} {explanation}"
            try:
                encodings = tokenizer(
                    text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self.input_ids.append(encodings["input_ids"].squeeze())
                self.attention_masks.append(encodings["attention_mask"].squeeze())
                self.labels.append(encodings["input_ids"].squeeze())
            except Exception as e:
                print(f"Error encoding row {idx}: {e}")
                continue
        
        if not self.input_ids:
            raise ValueError("Dataset is empty. No valid prompt-explanation pairs processed.")
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

def save_synthetic_data(data: List[Dict[str, str]], output_path: str) -> pd.DataFrame:
    """Save synthetic prompt-explanation pairs to a CSV file."""
    if not data:
        raise ValueError("Synthetic dataset is empty.")
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        df.to_csv(output_path, index=False)
        return df
    except Exception as e:
        raise Exception(f"Error saving synthetic data to {output_path}: {e}")

def load_synthetic_data() -> List[Dict[str, str]]:
    """Load or generate synthetic prompt-explanation pairs."""
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            return df[["prompt", "explanation"]].to_dict('records')
        except Exception as e:
            print(f"Error loading synthetic data from {DATA_PATH}: {e}")
    
    # Default synthetic data
    sample_data = [
        {
            "prompt": (
                "Explain why product B000123456 is recommended: Sentiment is Positive, "
                "users mentioned primary phrase 'easy setup' and secondary phrases "
                "'great connectivity', 'fast speed'."
            ),
            "explanation": (
                "Product B000123456 is recommended because users found it to have an easy setup, "
                "great connectivity, and fast speed, making it a reliable choice for a router or access point."
            ),
        },
        {
            "prompt": (
                "Explain why product B0015MCPKU is recommended: Sentiment is Positive, "
                "users mentioned primary phrase 'excellent quality' and secondary phrases "
                "'fast speed', 'great connectivity'."
            ),
            "explanation": (
                "Product B0015MCPKU is recommended because users found it to have excellent quality, "
                "fast speed, and great connectivity, making it a top choice for electronics."
            ),
        },
        {
            "prompt": (
                "Explain why product B004U78628 is recommended: Sentiment is Positive, "
                "users mentioned primary phrase 'great battery' and secondary phrases "
                "'great keyboard', 'easy setup'."
            ),
            "explanation": (
                "Product B004U78628 is recommended because users found it to have a great battery, "
                "great keyboard, and easy setup, offering a versatile user experience."
            ),
        },
        {
            "prompt": (
                "Explain why product B000213ZFE is recommended: Sentiment is Positive, "
                "users mentioned primary phrase 'easy installation' and secondary phrases "
                "'reliable wifi', 'fast speed'."
            ),
            "explanation": (
                "Product B000213ZFE is recommended because users found it to have an easy installation, "
                "reliable wifi, and fast speed, ideal for network devices."
            ),
        },
        {
            "prompt": (
                "Explain why product B002IT1BFO is recommended: Sentiment is Positive, "
                "users mentioned primary phrase 'great sound' and secondary phrases "
                "'comfortable headphones', 'long battery life'."
            ),
            "explanation": (
                "Product B002IT1BFO is recommended because users found it to have great sound, "
                "comfortable headphones, and long battery life, perfect for audio enthusiasts."
            ),
        },
        {
            "prompt": (
                "Explain why product B002IJA1EG is recommended: Sentiment is Positive, "
                "users mentioned primary phrase 'drive fast' and secondary phrases "
                "'large storage', 'reliable performance'."
            ),
            "explanation": (
                "Product B002IJA1EG is recommended because users found it to have a fast drive, "
                "large storage, and reliable performance, suitable for high-performance computing."
            ),
        },
        {
            "prompt": (
                "Explain why product B000789012 is recommended: Sentiment is Neutral, "
                "users mentioned primary phrase 'reliable wifi' and secondary phrases ''."
            ),
            "explanation": (
                "Product B000789012 is recommended because users found it to have reliable wifi, "
                "providing consistent performance for networking needs."
            ),
        },
    ]
    save_synthetic_data(sample_data, DATA_PATH)
    return sample_data

def fine_tune_gpt2() -> None:
    """Fine-tune GPT-2 for explanation generation."""
    # Initialize tokenizer and model
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
    if _model is None:
        _model = GPT2LMHeadModel.from_pretrained("gpt2")
        _model.config.pad_token_id = _tokenizer.eos_token_id
    
    # Load synthetic data
    sample_data = load_synthetic_data()
    df = pd.DataFrame(sample_data)
    
    # Prepare dataset
    try:
        dataset = ExplanationDataset(df, _tokenizer, max_length=512)
    except ValueError as e:
        raise ValueError(f"Failed to create dataset: {e}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=TRAINING_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        max_steps=len(dataset) * 3,  # Adjusted for small dataset
        report_to="none",
    )
    
    # Initialize and train
    trainer = Trainer(model=_model, args=training_args, train_dataset=dataset)
    try:
        trainer.train()
    except Exception as e:
        raise Exception(f"Training failed: {e}")
    
    # Save model
    os.makedirs(MODEL_PATH, exist_ok=True)
    _model.save_pretrained(MODEL_PATH)
    _tokenizer.save_pretrained(MODEL_PATH)

def generate_explanation(asin: str, sentiment: str, phrases: Dict[str, any]) -> str:
    """
    Generate a natural language explanation for a product recommendation.

    Args:
        asin: Product ASIN (e.g., 'B000123456').
        sentiment: Sentiment label ('Positive' or 'Neutral').
        phrases: Dictionary with 'primary' (str) and 'secondary' (list of str) phrases.

    Returns:
        Explanation string, e.g., "Product B000123456 is recommended because users found it to have
        an easy setup, great connectivity, and fast speed, making it a reliable choice for a router or access point."

    Raises:
        ValueError: If input arguments are invalid.
    """
    # Input validation
    if not isinstance(asin, str) or not asin.strip():
        raise ValueError(f"Invalid asin: {asin}. Must be a non-empty string.")
    if not isinstance(sentiment, str) or sentiment not in ['Positive', 'Neutral']:
        raise ValueError(f"Invalid sentiment: {sentiment}. Must be 'Positive' or 'Neutral'.")
    if not isinstance(phrases, dict) or 'primary' not in phrases or 'secondary' not in phrases:
        raise ValueError(f"Invalid phrases: {phrases}. Must be a dict with 'primary' and 'secondary' keys.")
    if not isinstance(phrases['primary'], str) or not phrases['primary'].strip():
        raise ValueError(f"Invalid primary phrase: {phrases['primary']}. Must be a non-empty string.")
    if not isinstance(phrases['secondary'], list) or not all(isinstance(p, str) and p.strip() for p in phrases['secondary']):
        phrases['secondary'] = [p for p in phrases['secondary'] if isinstance(p, str) and p.strip()]
    
    # Load or fine-tune model
    global _model, _tokenizer
    model_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    model_valid = all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in model_files)
    
    if _model is None or _tokenizer is None or not model_valid:
        if model_valid:
            try:
                _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
                _model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model from {MODEL_PATH}: {e}. Fine-tuning...")
                fine_tune_gpt2()
                _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
                _model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        else:
            fine_tune_gpt2()
            _tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
            _model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    _tokenizer.pad_token = _tokenizer.eos_token
    _model.config.pad_token_id = _tokenizer.eos_token_id
    _model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(device)
    
    # Construct prompt
    secondary_text = (
        f" and secondary phrases '{', '.join(phrases['secondary'])}'" if phrases['secondary'] else ""
    )
    prompt = (
        f"Explain why product {asin} is recommended: Sentiment is {sentiment}, "
        f"users mentioned primary phrase '{phrases['primary']}'{secondary_text}."
    )
    
    # Tokenize prompt
    try:
        inputs = _tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception as e:
        raise ValueError(f"Error tokenizing prompt: {e}")
    
    # Generate explanation
    try:
        outputs = _model.generate(
            **inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    except Exception as e:
        raise ValueError(f"Error generating explanation: {e}")
    
    # Validate and decode output
    if outputs.shape[0] < 1 or outputs.shape[1] < 1:
        raise ValueError(f"Invalid outputs shape: {outputs.shape}. Expected at least one sequence.")
    
    explanation = _tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
    explanation = explanation.replace(prompt, "").strip()
    explanation = re.sub(r'\s+', ' ', explanation).strip()
    
    # Format explanation
    if not explanation:
        explanation = f"users found it to have {phrases['primary']}{secondary_text}."
    if not explanation.startswith("Product"):
        explanation = f"Product {asin} is recommended because {explanation[0].lower() + explanation[1:]}"
    if not explanation.endswith("."):
        explanation += "."
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return explanation

def main():
    """Test the explanation generation system."""
    try:
        # Test fine-tuning
        if not all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in ['config.json', 'pytorch_model.bin']):
            print("Fine-tuning GPT-2 model...")
            fine_tune_gpt2()
        
        # Test explanation generation
        test_input = {
            "asin": "B000123456",
            "sentiment": "Positive",
            "phrases": {
                "primary": "easy setup",
                "secondary": ["great connectivity", "fast speed"]
            }
        }
        explanation = generate_explanation(
            test_input["asin"],
            test_input["sentiment"],
            test_input["phrases"]
        )
        print(f"Test Explanation: {explanation}")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()