import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.preprocess import preprocess_pipeline
from src.feature_engineering import create_features

def train_models(data_path, model_save_path="model.pkl"):
    # 1. Load
    print("Loading data...")
    df = load_data(data_path)
    
    # 2. Preprocess
    print("Preprocessing...")
    df, encoders = preprocess_pipeline(df)
    
    # 3. Feature Engineering
    print("Feature Engineering...")
    df = create_features(df)
    
    # 4. Split
    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Handle Imbalance
    print("Handling Class Imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # 6. Train Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        results[name] = score
        print(f"{name} Accuracy: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            
    # 7. Hyperparameter Tuning (on Random Forest as requested)
    print("Tuning Hyperparameters for Random Forest...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"Best Params: {grid_search.best_params_}")
    tuned_rf = grid_search.best_estimator_
    tuned_score = accuracy_score(y_test, tuned_rf.predict(X_test))
    print(f"Tuned Random Forest Accuracy: {tuned_score:.4f}")
    
    if tuned_score > best_score:
        best_model = tuned_rf
        
    # 8. Save Best Model
    print(f"Saving best model to {model_save_path}...")
    joblib.dump(best_model, model_save_path)
    
    # Save encoders
    print("Saving encoders to encoders.pkl...")
    joblib.dump(encoders, "encoders.pkl")
    
    # Save test data for evaluation
    joblib.dump((X_test, y_test), "test_data.pkl")
    
    return results

if __name__ == "__main__":
    train_models("data/Hotel Reservations.csv")
