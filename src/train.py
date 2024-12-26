import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import argparse
import yaml
import sys

def train_model(input_csv, model_output):
    try:
        # Read parameters from params.yaml
        with open("params.yaml", 'r') as params_file:
            params = yaml.safe_load(params_file)
        
        # Get all model parameters
        model_params = params['train']['model_params']
        print(f"Training model with parameters: {model_params}")
        
        # Read the processed CSV
        df = pd.read_csv(input_csv)
        if df.empty:
            raise ValueError("The input CSV file is empty")
            
        X = df.drop("species", axis=1)
        y = df["species"]

        # Create and train model
        model = LogisticRegression(**model_params)
        model.fit(X, y)

        # Save model
        with open(model_output, "wb") as f:
            pickle.dump(model, f)
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the processed CSV")
    parser.add_argument("--model", required=True, help="Path to save the model file")
    args = parser.parse_args()

    train_model(args.input, args.model)