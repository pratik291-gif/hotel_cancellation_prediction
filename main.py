import argparse
import sys
from src.eda import perform_eda
from src.train import train_models
from src.evaluate import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Hotel Booking Cancellation Prediction Pipeline")
    parser.add_argument('--step', type=str, choices=['eda', 'train', 'evaluate', 'all'], default='all', help='Pipeline step to run')
    parser.add_argument('--data_path', type=str, default="data/Hotel Reservations.csv", help='Path to dataset')
    
    args = parser.parse_args()
    
    try:
        if args.step in ['eda', 'all']:
            print("Running EDA...")
            perform_eda(args.data_path)
            
        if args.step in ['train', 'all']:
            print("Running Training...")
            train_models(args.data_path)
            
        if args.step in ['evaluate', 'all']:
            print("Running Evaluation...")
            evaluate_model()
            
        print("Pipeline completed successfully.")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
