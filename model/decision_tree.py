#!/usr/bin/env python3
"""
Decision Tree Classifier Model Implementation for Wine Quality Prediction
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
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

class DecisionTreeModel:
    """Decision Tree Classifier Model for Wine Quality Prediction"""
    
    def __init__(self, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.model = DecisionTreeClassifier(
            random_state=random_state, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.is_trained = False
        
    def train(self, X_train, X_test, y_train, y_test):
        """Train the decision tree model"""
        # Train model (no scaling needed for tree-based models)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
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
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.feature_importances_
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']

if __name__ == "__main__":
    # Example usage
    print("Decision Tree Classifier Model for Wine Quality Prediction")
    print("This module provides a complete implementation of decision tree classification")
    print("with training, evaluation, feature importance, and persistence capabilities.")
