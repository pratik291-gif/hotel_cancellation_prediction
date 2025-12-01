import pandas as pd
import os

def load_data(filepath):
    """
    Load the dataset from the specified filepath.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe with standardized column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Standardize column names to match the assignment requirements where possible
    column_mapping = {
        'no_of_weekend_nights': 'stays_in_weekend_nights',
        'no_of_week_nights': 'stays_in_week_nights',
        'no_of_adults': 'adults',
        'no_of_children': 'children',
        'avg_price_per_room': 'adr',
        'booking_status': 'booking_status_str' # Keep original for now, process later or rename
    }
    
    df = df.rename(columns=column_mapping)
    
    # Handle target variable mapping immediately for convenience
    # 'Canceled' -> 1, 'Not_Canceled' -> 0
    if 'booking_status_str' in df.columns:
        df['is_canceled'] = df['booking_status_str'].apply(lambda x: 1 if x == 'Canceled' else 0)
        df = df.drop(columns=['booking_status_str'])
        
    return df

if __name__ == "__main__":
    # Test loading
    try:
        df = load_data("hotel_cancellation_prediction/data/Hotel Reservations.csv")
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
