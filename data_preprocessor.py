#!/usr/bin/env python3
"""
data_preprocessor.py - AI Bug Bounty Data Preprocessor

This module provides functionality to clean, preprocess, and vectorize text data 
for training machine learning models.
"""

import re
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Import logging utilities
try:
    from utils import setup_logger, log_info, log_error
except ImportError:
    import logging
    def setup_logger(name): return logging.getLogger(name)
    def log_info(msg): logging.info(msg)
    def log_error(msg): logging.error(msg)

class DataPreprocessor:
    """Main class for preprocessing data for the AI model."""
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the preprocessor with a configuration."""
        self.config = config or self._get_default_config()
        self.logger = setup_logger(__name__)
        self.tokenizer = None
        self.vectorizer = None
        log_info(f"DataPreprocessor initialized with config: {self.config}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Return a default configuration dictionary."""
        return {
            "max_features": 20000,
            "max_len": 256,
            "embedding_dim": 128,
            "test_size": 0.2,
            "random_state": 42,
            "vectorization_strategy": "keras_tokenizer" # or "tfidf"
        }

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
        text = re.sub(r'<.*?>', '', text) # Remove HTML tags
        text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
        return text

    def load_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Load data from a list of dictionaries into a pandas DataFrame."""
        df = pd.DataFrame(data)
        self.logger.info(f"Loaded {len(df)} records into DataFrame.")
        return df

    def preprocess_data(self, data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Main method to preprocess raw data.
        Cleans, tokenizes, and splits the data into training and testing sets.
        """
        self.logger.info("Starting data preprocessing...")
        try:
            df = self.load_data(data)
            
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Input data must contain 'text' and 'label' columns.")

            df['cleaned_text'] = df['text'].apply(self.clean_text)
            
            X = df['cleaned_text'].values
            y = df['label'].values # Assuming labels are already encoded

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'], 
                random_state=self.config['random_state']
            )

            strategy = self.config.get('vectorization_strategy', 'keras_tokenizer')
            if strategy == 'tfidf':
                X_train_vec, X_test_vec = self._vectorize_tfidf(X_train, X_test)
            elif strategy == 'keras_tokenizer':
                X_train_vec, X_test_vec = self._vectorize_keras(X_train, X_test)
            else:
                raise ValueError(f"Invalid vectorization strategy: {strategy}")

            self.logger.info("Data preprocessing completed successfully.")
            
            vocab_size = len(self.tokenizer.word_index) + 1 if strategy == 'keras_tokenizer' else len(self.vectorizer.vocabulary_)

            return {
                'X_train': X_train_vec,
                'X_test': X_test_vec,
                'y_train': y_train,
                'y_test': y_test,
                'tokenizer': self.tokenizer if strategy == 'keras_tokenizer' else self.vectorizer,
                'vocab_size': vocab_size,
                'max_len': self.config['max_len']
            }
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {e}")
            raise

    def _vectorize_tfidf(self, X_train, X_test):
        """Vectorize text data using TF-IDF."""
        self.vectorizer = TfidfVectorizer(max_features=self.config['max_features'])
        X_train_vec = self.vectorizer.fit_transform(X_train).toarray()
        X_test_vec = self.vectorizer.transform(X_test).toarray()
        return X_train_vec, X_test_vec

    def _vectorize_keras(self, X_train, X_test):
        """Tokenize and pad text data using Keras Tokenizer."""
        self.tokenizer = Tokenizer(num_words=self.config['max_features'], oov_token="<OOV>")
        self.tokenizer.fit_on_texts(X_train)
        
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.config['max_len'], padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.config['max_len'], padding='post', truncating='post')
        
        return X_train_pad, X_test_pad

# Convenience function for connector.py
def preprocess_data(item: Dict, **kwargs) -> Any:
    """
    Convenience function to instantiate and run DataPreprocessor.
    'item' should contain the data to be processed under a 'data' key.
    """
    config = item.get('config', {})
    data_to_process = item.get('data')

    if not data_to_process:
        raise ValueError("Checklist item for preprocessing must contain a 'data' key.")

    preprocessor = DataPreprocessor(config=config)
    return preprocessor.preprocess_data(data_to_process, **kwargs)

if __name__ == '__main__':
    # Example Usage
    log_info("Running DataPreprocessor example...")
    
    # Sample data
    sample_data = [
        {'text': 'This is a test sentence for our model.', 'label': 0},
        {'text': 'Another test sentence with different words.', 'label': 1},
        {'text': '<p>HTML content to be cleaned</p>', 'label': 0},
        {'text': 'Visit http://example.com for more info.', 'label': 1},
        {'text': 'Final test with many many words to check padding.', 'label': 0}
    ]
    
    # Create an item dictionary as expected by the connector
    item = {
        'type': 'data_preprocessing',
        'data': sample_data,
        'config': {
            "vectorization_strategy": "keras_tokenizer",
            "max_len": 10
        }
    }
    
    try:
        results = preprocess_data(item)
        log_info("Preprocessing successful!")
        log_info(f"Train data shape: {results['X_train'].shape}")
        log_info(f"Test data shape: {results['X_test'].shape}")
        log_info(f"Vocab size: {results['vocab_size']}")
    except Exception as e:
        log_error(f"Example failed: {e}")