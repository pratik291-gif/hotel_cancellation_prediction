import pandas as pd
import joblib
from src.preprocess import preprocess_pipeline
from src.feature_engineering import create_features

def load_artifacts(model_path="model.pkl", encoders_path="encoders.pkl"):
    """
    Load trained model and encoders.
    """
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders

def predict_cancellation(data, model, encoders):
    """
    Predict if a booking will be canceled.
    
    Args:
        data (dict or pd.DataFrame): Input data.
        model: Trained model.
        encoders: Fitted encoders.
        
    Returns:
        np.array: Prediction (0 or 1)
        np.array: Probability
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()
        
    # Preprocess
    # Note: We must ensure columns match training data expectations
    # The pipeline handles cleaning, outliers, and encoding
    df_processed, _ = preprocess_pipeline(df, encoders=encoders)
    
    # Feature Engineering
    df_fe = create_features(df_processed)
    
    # Ensure column order matches model (if necessary, though sklearn usually handles it if names match)
    # Ideally we should save feature names during training to be safe, but for now we rely on consistency
    
    # Predict
    prediction = model.predict(df_fe)
    probability = model.predict_proba(df_fe)[:, 1]
    
    return prediction, probability

if __name__ == "__main__":
    # Example usage
    print("Loading artifacts...")
    model, encoders = load_artifacts()
    
    # Sample input (taken from dataset head)
    # INN00002,2,0,2,3,Not Selected,0,Room_Type 1,5,2018,11,6,Online,0,0,0,106.68,1,Not_Canceled
    sample_input = {
        'no_of_adults': 2,
        'no_of_children': 0,
        'no_of_weekend_nights': 2,
        'no_of_week_nights': 3,
        'type_of_meal_plan': 'Not Selected',
        'required_car_parking_space': 0,
        'room_type_reserved': 'Room_Type 1',
        'lead_time': 5,
        'arrival_year': 2018,
        'arrival_month': 11,
        'arrival_date': 6,
        'market_segment_type': 'Online',
        'repeated_guest': 0,
        'no_of_previous_cancellations': 0,
        'no_of_previous_bookings_not_canceled': 0,
        'avg_price_per_room': 106.68,
        'no_of_special_requests': 1
    }
    
    # We need to rename keys to match what data_loader does, OR update data_loader to be reusable
    # The data_loader renames columns. In production, input might come with original names or new names.
    # Let's assume input comes as raw data (original names) and we need to map them.
    # We should probably expose the mapping from data_loader or just do it here.
    
    # Mapping from data_loader.py
    column_mapping = {
        'no_of_weekend_nights': 'stays_in_weekend_nights',
        'no_of_week_nights': 'stays_in_week_nights',
        'no_of_adults': 'adults',
        'no_of_children': 'children',
        'avg_price_per_room': 'adr'
    }
    
    # Apply mapping
    sample_df = pd.DataFrame([sample_input])
    sample_df = sample_df.rename(columns=column_mapping)
    
    print("\nPredicting for sample input:")
    print(sample_df.iloc[0].to_dict())
    
    pred, prob = predict_cancellation(sample_df, model, encoders)
    
    print(f"\nPrediction: {'Canceled' if pred[0] == 1 else 'Not Canceled'}")
    print(f"Probability of Cancellation: {prob[0]:.4f}")