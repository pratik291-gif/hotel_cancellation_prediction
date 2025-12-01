import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    Handle missing values and duplicates.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Handle missing values (if any appeared after loading, though EDA showed none)
    # For numeric, fill with median; for categorical, mode.
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    return df

def handle_outliers(df, columns, method='capping', factor=1.5):
    """
    Handle outliers in specified columns using IQR.
    """
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        if method == 'capping':
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        elif method == 'removal':
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
    return df_clean

def encode_categorical(df, encoders=None):
    """
    Encode categorical variables using Label Encoding.
    Returns the dataframe and the encoders.
    """
    df_encoded = df.copy()
    
    if encoders is None:
        encoders = {}
        is_training = True
    else:
        is_training = False
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        # Skip Booking_ID if present, it's not a feature
        if col == 'Booking_ID':
            continue
            
        if is_training:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
        else:
            if col in encoders:
                le = encoders[col]
                # Handle unseen labels by mapping them to a known label or handling error
                # For simplicity, we'll use a safe transform approach:
                # Map unseen labels to the first class (0) or handle exception
                # Here we strictly transform, assuming data consistency or catching errors
                try:
                    df_encoded[col] = le.transform(df_encoded[col])
                except ValueError:
                    # Fallback for unseen labels: map to most frequent (mode) or 0
                    # This is a simple hack for production stability
                    # Get classes
                    classes = list(le.classes_)
                    # Map unknown to the first class
                    df_encoded[col] = df_encoded[col].apply(lambda x: x if x in classes else classes[0])
                    df_encoded[col] = le.transform(df_encoded[col])
            
    return df_encoded, encoders

def preprocess_pipeline(df, encoders=None):
    """
    Full preprocessing pipeline.
    """
    # 1. Clean
    df = clean_data(df)
    
    # 2. Outliers
    # Handle outliers in 'adr' and 'lead_time'
    df = handle_outliers(df, columns=['adr', 'lead_time'], method='capping')
    
    # 3. Encode
    df, encoders = encode_categorical(df, encoders=encoders)
    
    # Drop Booking_ID if it exists
    if 'Booking_ID' in df.columns:
        df = df.drop(columns=['Booking_ID'])
        
    return df, encoders
