import pandas as pd
import numpy as np

def create_features(df):
    """
    Create new features as per requirements.
    """
    df_fe = df.copy()
    
    # 1. Total stay nights
    df_fe['total_stay_nights'] = df_fe['stays_in_weekend_nights'] + df_fe['stays_in_week_nights']
    
    # 2. Total guests
    df_fe['total_guests'] = df_fe['adults'] + df_fe['children']
    
    # 3. Booking lead time category
    # Short: < 7 days, Medium: 7-30 days, Long: > 30 days
    bins = [-1, 7, 30, float('inf')]
    labels = [0, 1, 2] # Encoded as 0, 1, 2 for Short, Medium, Long
    df_fe['lead_time_category'] = pd.cut(df_fe['lead_time'], bins=bins, labels=labels).astype(int)
    
    # 4. Average ADR per person
    # Avoid division by zero
    df_fe['adr_per_person'] = df_fe['adr'] / (df_fe['total_guests'] + 1e-5)
    
    # 5. Weekend booking flag
    df_fe['is_weekend'] = (df_fe['stays_in_weekend_nights'] > 0).astype(int)
    
    return df_fe
