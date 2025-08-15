"""
AI Trainer Module for Transformer Model Training and Evaluation

This module provides functionality to train and evaluate transformer models
with comprehensive metrics and model persistence capabilities.
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

# Import custom modules
try:
    from ai_model import TransformerModel
    from data_preprocessor import DataPreprocessor
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure ai_model.py and data_preprocessor.py are in the same directory")


class AITrainer:
    """
    Trainer class for transformer models with comprehensive evaluation metrics.
    """
    
    def __init__(self, model_config: Dict[str, Any], device: str = None):
        """
        Initialize the trainer.
        
        Args:
            model_config: Dictionary containing model configuration parameters
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Create models directory if it doesn't exist
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_model(self, vocab_size: int, num_classes: int):
        """
        Initialize the transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            num_classes: Number of output classes
        """
        try:
            self.model = TransformerModel(
                vocab_size=vocab_size,
                num_classes=num_classes,
                **self.model_config
            ).to(self.device)
            
            self.logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
            self.logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
            
    def setup_training(self, learning_rate: float = 0.001, weight_decay: float = 0.01):
        """
        Setup optimizer and loss criterion.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        if self.model is None:
            raise ValueError("Model must be initialized before setting up training")
            
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.logger.info(f"Training setup complete - LR: {learning_rate}, Weight Decay: {weight_decay}")
        
    def create_data_loaders(self, train_data: Tuple[np.ndarray, np.ndarray], 
                          val_data: Tuple[np.ndarray, np.ndarray],
                          batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders from preprocessed data.
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        self.logger.info(f"Created data loaders - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                self.logger.info(f'Train Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.6f}')
                
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (validation_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
        
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing various metrics
        """
        # Calculate basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate macro averages as well
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
        }
        
        return metrics
        
    def train(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray],
              epochs: int = 10,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              weight_decay: float = 0.01,
              save_best: bool = True,
              patience: int = 5) -> Dict[str, Any]:
        """
        Complete training loop with validation and early stopping.
        
        Args:
            train_data: Training data tuple (X, y)
            val_data: Validation data tuple (X, y)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            save_best: Whether to save the best model
            patience: Early stopping patience
            
        Returns:
            Dictionary containing training history
        """
        # Setup training
        if self.model is None:
            vocab_size = np.max(train_data[0]) + 1
            num_classes = len(np.unique(train_data[1]))
            self.initialize_model(vocab_size, num_classes)
            
        self.setup_training(learning_rate, weight_decay)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_data, val_data, batch_size
        )
        
        # Training variables
        best_val_loss = float('inf')
        best_f1 = 0.0
        patience_counter = 0
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'metrics': [],
            'best_epoch': 0
        }
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, metrics = self.validate(val_loader)
            
            # Log progress
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"F1: {metrics['f1_weighted']:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Store history
            training_history['train_losses'].append(train_loss)
            training_history['val_losses'].append(val_loss)
            training_history['metrics'].append(metrics)
            
            # Save best model
            if save_best and (val_loss < best_val_loss or metrics['f1_weighted'] > best_f1):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if metrics['f1_weighted'] > best_f1:
                    best_f1 = metrics['f1_weighted']
                    
                training_history['best_epoch'] = epoch
                self.save_model(f"best_model_epoch_{epoch+1}.pth", metrics)
                patience_counter = 0
                self.logger.info(f"New best model saved at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
                
        total_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"Training completed in {total_time/60:.2f} minutes")
        
        return training_history
        
    def evaluate_model(self, test_data: Tuple[np.ndarray, np.ndarray],
                      batch_size: int = 32) -> Dict[str, Any]:
        """
        Comprehensive model evaluation on test data.
        
        Args:
            test_data: Test data tuple (X, y)
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing detailed evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        X_test, y_test = test_data
        
        # Create test loader
        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate
        test_loss, metrics = self.validate(test_loader)
        
        # Get detailed predictions for confusion matrix
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        evaluation_results = {
            'test_loss': test_loss,
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'true_labels': all_targets
        }
        
        # Log results
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("=" * 50)
        self.logger.info(f"Test Loss: {test_loss:.4f}")
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        self.logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
        self.logger.info(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        self.logger.info(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
        
        return evaluation_results
        
    def save_model(self, filename: str, metrics: Dict[str, float] = None):
        """
        Save the trained model and associated metadata.
        
        Args:
            filename: Name of the file to save
            metrics: Optional metrics to save with the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        model_path = os.path.join(self.models_dir, filename)
        
        # Prepare save dict
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat(),
        }
        
        if metrics:
            save_dict['metrics'] = metrics
            
        # Save model
        torch.save(save_dict, model_path)
        
        # Save model config as JSON for easy reference
        config_path = model_path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(save_dict.get('metrics', {}), f, indent=2)
            
        self.logger.info(f"Model saved to {model_path}")
        
    def load_model(self, filename: str, vocab_size: int, num_classes: int):
        """
        Load a previously saved model.
        
        Args:
            filename: Name of the model file to load
            vocab_size: Vocabulary size for model initialization
            num_classes: Number of classes for model initialization
        """
        model_path = os.path.join(self.models_dir, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with saved config
        self.model_config = checkpoint['model_config']
        self.initialize_model(vocab_size, num_classes)
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Restore training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f"Model loaded from {model_path}")


def main():
    """
    Example usage of the AITrainer class.
    """
    # Example configuration
    model_config = {
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 1024,
        'max_seq_len': 512,
        'dropout': 0.1
    }
    
    # Initialize trainer
    trainer = AITrainer(model_config)
    
    try:
        # Initialize data preprocessor
        preprocessor = DataPreprocessor()
        
        # Example: Load and preprocess data
        # train_data, val_data, test_data = preprocessor.load_and_preprocess_data("your_data.csv")
        
        # Example training (with dummy data)
        print("AITrainer initialized successfully!")
        print(f"Using device: {trainer.device}")
        print("Ready for training with real data.")
        
    except NameError:
        print("DataPreprocessor not available. Please ensure data_preprocessor.py is implemented.")
        print("AITrainer is ready to use with preprocessed data.")


if __name__ == "__main__":
    main()