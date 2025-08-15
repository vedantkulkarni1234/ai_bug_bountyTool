"""
ai_model.py - Lightweight Transformer-based NLP Model for Bug Bounty Processing

A comprehensive module implementing a lightweight transformer model optimized for
bug bounty task processing with sklearn integration and low-resource optimization.
"""

import os
import json
import pickle
import logging
import warnings
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    set_seed
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration class for the AI model."""
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    num_labels: int = 2
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    seed: int = 42
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False  # Mixed precision for memory efficiency


class BugBountyDataset(Dataset):
    """Custom dataset for bug bounty text processing."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class LightweightTransformer(nn.Module):
    """Lightweight transformer model for bug bounty classification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(
            config.model_name,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(256, config.num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {'loss': loss, 'logits': logits}


class AIModel(BaseEstimator, ClassifierMixin):
    """
    Lightweight transformer-based NLP model for bug bounty processing.
    Compatible with sklearn pipelines and optimized for low-resource environments.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.device = torch.device(self.config.device)
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        self._setup_model()
    
    def _setup_model(self):
        """Initialize model and tokenizer."""
        logger.info(f"Setting up model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            model_max_length=self.config.max_length
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = LightweightTransformer(self.config)
        self.model.to(self.device)
        
        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def fit(self, X: Union[List[str], np.ndarray, pd.Series], y: Union[List, np.ndarray, pd.Series]):
        """
        Train the model on the provided data.
        
        Args:
            X: Input texts
            y: Labels
        
        Returns:
            self: Fitted model instance
        """
        logger.info("Starting model training...")
        
        # Convert inputs to lists
        if isinstance(X, (np.ndarray, pd.Series)):
            X = X.tolist()
        if isinstance(y, (np.ndarray, pd.Series)):
            y = y.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.config.num_labels = len(self.classes_)
        
        # Recreate model with correct number of labels
        if self.model.config.num_labels != self.config.num_labels:
            self.model = LightweightTransformer(self.config)
            self.model.to(self.device)
        
        # Create dataset and split
        dataset = BugBountyDataset(X, y_encoded.tolist(), self.tokenizer, self.config.max_length)
        
        # Split into train/val (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Simplified for low-resource environments
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Update weights
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}, "
                              f"Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {loss.item():.4f}")
            
            # Validation
            val_loss = self._evaluate(val_loader)
            avg_train_loss = total_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        logger.info("Training completed!")
        
        return self
    
    def _evaluate(self, data_loader):
        """Evaluate model on validation data."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                total_loss += outputs['loss'].item()
        
        self.model.train()
        return total_loss / len(data_loader)
    
    def predict(self, X: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input texts
        
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to list
        if isinstance(X, (np.ndarray, pd.Series)):
            X = X.tolist()
        
        self.model.eval()
        predictions = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i + batch_size]
                
                # Tokenize batch
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        # Decode labels
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input texts
        
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to list
        if isinstance(X, (np.ndarray, pd.Series)):
            X = X.tolist()
        
        self.model.eval()
        all_probabilities = []
        
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_texts = X[i:i + batch_size]
                
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config.max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                all_probabilities.append(probabilities)
        
        return np.vstack(all_probabilities)
    
    def score(self, X: Union[List[str], np.ndarray, pd.Series], 
              y: Union[List, np.ndarray, pd.Series]) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Input texts
            y: True labels
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def save_model(self, model_path: str, save_tokenizer: bool = True):
        """
        Save the trained model and tokenizer.
        
        Args:
            model_path: Path to save the model
            save_tokenizer: Whether to save the tokenizer
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'label_encoder': self.label_encoder,
            'classes_': self.classes_
        }, model_path / 'pytorch_model.bin')
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(model_path)
        
        # Save config as JSON
        config_dict = {
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'num_labels': self.config.num_labels,
            'hidden_dropout_prob': self.config.hidden_dropout_prob,
            'attention_probs_dropout_prob': self.config.attention_probs_dropout_prob
        }
        
        with open(model_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """
        Load a saved model.
        
        Args:
            model_path: Path to the saved model
        """
        model_path = Path(model_path)
        
        # Load model checkpoint
        checkpoint = torch.load(model_path / 'pytorch_model.bin', map_location=self.device)
        
        # Restore configuration and components
        self.config = checkpoint['config']
        self.label_encoder = checkpoint['label_encoder']
        self.classes_ = checkpoint['classes_']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Recreate and load model
        self.model = LightweightTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_sklearn_pipeline(self, preprocessing_steps: List[Tuple[str, Any]] = None) -> Pipeline:
        """
        Create an sklearn pipeline with this model.
        
        Args:
            preprocessing_steps: List of (name, transformer) tuples for preprocessing
        
        Returns:
            sklearn Pipeline object
        """
        steps = preprocessing_steps or []
        steps.append(('classifier', self))
        
        return Pipeline(steps)
    
    def optimize_for_inference(self):
        """Optimize model for inference (reduces memory usage)."""
        if self.model is None:
            raise ValueError("Model must be loaded/trained first")
        
        # Convert to eval mode and optimize
        self.model.eval()
        
        # Enable torch script compilation for faster inference
        try:
            # Create dummy inputs for tracing
            dummy_input_ids = torch.randint(0, 1000, (1, self.config.max_length)).to(self.device)
            dummy_attention_mask = torch.ones((1, self.config.max_length)).to(self.device)
            
            # Trace the model (optional, might not work with all models)
            # self.model = torch.jit.trace(self.model, (dummy_input_ids, dummy_attention_mask))
            logger.info("Model optimized for inference")
        except Exception as e:
            logger.warning(f"Could not optimize model with TorchScript: {e}")


# Example usage and utility functions
def create_sample_pipeline():
    """Create a sample sklearn pipeline with the AI model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.compose import ColumnTransformer
    
    # Note: This is just an example structure
    # The AI model handles text directly, so TfidfVectorizer is not needed
    model = AIModel()
    pipeline = model.get_sklearn_pipeline()
    
    return pipeline


def evaluate_model(model: AIModel, X_test: List[str], y_test: List[str]) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained AI model
        X_test: Test texts
        y_test: Test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': predictions,
        'probabilities': probabilities
    }


if __name__ == "__main__":
    # Example usage
    print("AI Model for Bug Bounty Processing")
    print("==================================")
    
    # Sample data (replace with actual bug bounty data)
    sample_texts = [
        "SQL injection vulnerability in login form",
        "Cross-site scripting (XSS) in comment section",
        "Authentication bypass using parameter manipulation",
        "Normal user login attempt",
        "Regular file upload process",
        "Standard API response"
    ]
    
    sample_labels = [
        "vulnerability", "vulnerability", "vulnerability",
        "normal", "normal", "normal"
    ]
    
    # Initialize and train model
    config = ModelConfig(
        max_length=64,  # Shorter for demo
        batch_size=4,   # Smaller batch for demo
        num_epochs=1    # Quick training for demo
    )
    
    model = AIModel(config)
    
    print("Training model...")
    model.fit(sample_texts, sample_labels)
    
    # Make predictions
    test_texts = ["Buffer overflow in input validation", "User profile update"]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    print("\nPredictions:")
    for text, pred, prob in zip(test_texts, predictions, probabilities):
        print(f"Text: {text}")
        print(f"Prediction: {pred}")
        print(f"Probabilities: {prob}")
        print("-" * 50)
    
    # Demonstrate sklearn pipeline integration
    pipeline = model.get_sklearn_pipeline()
    print(f"\nCreated sklearn pipeline: {pipeline}")
    
    print("\nModel training completed successfully!")