#!/usr/bin/env python3
"""
XGBoost Classifier Model Implementation for Wine Quality Prediction
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
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

class XGBoostModel:
    """XGBoost Classifier Model for Wine Quality Prediction"""
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.3, 
                 subsample=1.0, colsample_bytree=1.0, random_state=42, 
                 eval_metric='mlogloss', verbosity=0):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric=eval_metric,
            verbosity=verbosity
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def train(self, X_train, X_test, y_train, y_test):
        """Train the XGBoost model"""
        # Encode labels for XGBoost (required for multiclass)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model
        self.model.fit(X_train, y_train_encoded)
        self.is_trained = True
        
        # Make predictions
        y_pred_encoded = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Convert back to original labels
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
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
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, importance_type='weight'):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        return self.model.get_booster().get_score(importance_type=importance_type)
    
    def get_booster(self):
        """Get the underlying XGBoost booster"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting booster")
        return self.model.get_booster()
    
    def plot_importance(self, importance_type='weight', max_num_features=None):
        """Plot feature importance"""
        try:
            from xgboost import plot_importance
            if not self.is_trained:
                raise ValueError("Model must be trained before plotting importance")
            return plot_importance(
                self.model, 
                importance_type=importance_type,
                max_num_features=max_num_features
            )
        except ImportError:
            print("matplotlib is required for plotting feature importance")
            return None
    
    def save_model(self, filepath):
        """Save the trained model and label encoder"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load a saved model and label encoder"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']

if __name__ == "__main__":
    # Example usage
    print("XGBoost Classifier Model for Wine Quality Prediction")
    print("This module provides a complete implementation of XGBoost ensemble")
    print("with gradient boosting, feature importance, and persistence capabilities.")
