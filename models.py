#!/usr/bin/env python3
"""
Model Definitions for PPMI Classification

Includes:
- Classical ML models (Logistic Regression, Random Forest, XGBoost)
- Deep learning models (MLP)
- Sequence models (LSTM, GRU, 1D CNN)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Deep learning models will be skipped.")


class BaselineMLModels:
    """Wrapper for classical ML models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
    
    def get_models(self):
        """Get dictionary of model instances."""
        return {
            'logistic_regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                eval_metric='mlogloss',
                use_label_encoder=False
            ),
            'svm_linear': SVC(
                kernel='linear',
                probability=True,
                random_state=42,
                class_weight='balanced'
            ),
            'svm_rbf': SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        }
    
    def fit(self, X, y, model_name='all'):
        """Fit models to data."""
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.label_encoders[model_name] = le
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[model_name] = scaler
        
        models_to_fit = self.get_models() if model_name == 'all' else {model_name: self.get_models()[model_name]}
        
        for name, model in models_to_fit.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y_encoded)
            self.models[name] = model
    
    def predict(self, X, model_name):
        """Predict with a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X_scaled = self.scalers.get(model_name, StandardScaler()).transform(X)
        predictions = self.models[model_name].predict(X_scaled)
        le = self.label_encoders.get(model_name, LabelEncoder())
        return le.inverse_transform(predictions)
    
    def predict_proba(self, X, model_name):
        """Predict probabilities with a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        X_scaled = self.scalers.get(model_name, StandardScaler()).transform(X)
        return self.models[model_name].predict_proba(X_scaled)


if TORCH_AVAILABLE:
    class MLP(nn.Module):
        """Multi-layer perceptron for tabular data."""
        
        def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=3, dropout=0.3):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    
    class LSTMClassifier(nn.Module):
        """LSTM-based sequence classifier."""
        
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
            self.fc = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, (h_n, c_n) = self.lstm(x)
            # Use last hidden state
            last_hidden = h_n[-1]
            output = self.fc(last_hidden)
            return output
    
    
    class CNN1DClassifier(nn.Module):
        """1D CNN for sequence classification."""
        
        def __init__(self, input_dim, num_filters=64, kernel_size=3, num_classes=3, dropout=0.3):
            super(CNN1DClassifier, self).__init__()
            self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size, padding=1)
            self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_filters * 2, num_classes)
        
        def forward(self, x):
            # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            x = self.dropout(x)
            output = self.fc(x)
            return output


