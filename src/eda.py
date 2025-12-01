import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import load_data

def perform_eda(filepath, output_dir="eda_outputs"):
    """
    Perform EDA and save plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = load_data(filepath)
    
    # 1. Summary
    print("Dataset Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    
    # Save summary to text file
    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")
        f.write(f"Missing Values:\n{df.isnull().sum()}\n\n")
        f.write(f"Duplicates: {df.duplicated().sum()}\n")

    # 2. Visualizations
    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_canceled', data=df)
    plt.title("Booking Cancellation Distribution")
    plt.savefig(f"{output_dir}/cancellation_dist.png")
    plt.close()
    
    # Lead time distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['lead_time'], bins=50, kde=True)
    plt.title("Lead Time Distribution")
    plt.savefig(f"{output_dir}/lead_time_dist.png")
    plt.close()
    
    # ADR distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['adr'], bins=50, kde=True)
    plt.title("ADR Distribution")
    plt.savefig(f"{output_dir}/adr_dist.png")
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    print(f"EDA completed. Outputs saved to {output_dir}")

if __name__ == "__main__":
    perform_eda("data/Hotel Reservations.csv")
