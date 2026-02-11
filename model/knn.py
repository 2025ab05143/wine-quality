#!/usr/bin/env python3
"""
K-Nearest Neighbors Classifier Model Implementation for Wine Quality Prediction
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                           recall_score, f1_score, matthews_corrcoef)
from sklearn.preprocessing import LabelBinarizer
import pickle

def calculate_multiclass_auc(y_true, y_pred_proba):
    """Calculate AUC for multiclass classification using macro averaging"""
    try:
        lb = LabelBinarizer()
        y_true_bin = lb.fit_transform(y_true)
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        return roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
    except:
        return 0.0

class KNearestNeighborsModel:
    """K-Nearest Neighbors Classifier Model for Wine Quality Prediction"""
    
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', metric='minkowski'):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            metric=metric
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X_train, X_test, y_train, y_test):
        """Train the KNN model"""
        # Scale features (important for KNN)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC': calculate_multiclass_auc(y_test, y_pred_proba),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1': f1_score(y_test, y_pred, average='weighted'),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        
        return metrics, y_pred, y_pred_proba
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_neighbors(self, X, n_neighbors=None):
        """Get the k-nearest neighbors for given samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting neighbors")
        
        X_scaled = self.scaler.transform(X)
        
        if n_neighbors is None:
            n_neighbors = self.model.n_neighbors
            
        distances, indices = self.model.kneighbors(X_scaled, n_neighbors=n_neighbors)
        return distances, indices
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a saved model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']

if __name__ == "__main__":
    # Example usage
    print("K-Nearest Neighbors Classifier Model for Wine Quality Prediction")
    print("This module provides a complete implementation of KNN classification")
    print("with feature scaling, training, evaluation, and persistence capabilities.")
