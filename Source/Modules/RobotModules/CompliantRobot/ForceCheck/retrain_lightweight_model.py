#!/usr/bin/env python3
"""
Retrain the perturbation classifier with fewer trees for faster inference.
This creates a lightweight model optimized for real-time performance.
"""

import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def retrain_lightweight_model():
    """Load existing model, retrain with fewer trees, and save."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'models')
    
    # Load existing model to get training data (if you saved it)
    # Otherwise, you'll need to load your original training data
    
    print("To retrain the model with fewer trees:")
    print("1. Load your original training data (features and labels)")
    print("2. Train a new RandomForestClassifier with n_estimators=20 (instead of 100+)")
    print("3. Save the new model")
    print()
    print("Example code:")
    print("""
# Load your training data
# X_train, y_train = load_your_data()

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Train lightweight model (20 trees instead of 100)
model = RandomForestClassifier(
    n_estimators=20,        # Reduced from 100 for speed
    max_depth=10,           # Limit depth to prevent overfitting
    min_samples_split=20,   # Require more samples to split
    min_samples_leaf=10,    # Require more samples in leaves
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print(f"Lightweight model accuracy: {accuracy:.2%}")

# Save the model
with open('models/perturbation_classifier_rf.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Lightweight model saved!")
print(f"Trees: {len(model.estimators_)}")
print("This should be 5-10x faster than the original model.")
""")

if __name__ == "__main__":
    retrain_lightweight_model()
