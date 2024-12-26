import pandas as pd
import argparse
import numpy as np

def preprocess(input_csv, output_csv, add_noise=False, remove_samples=False):
    # Read the data
    df = pd.read_csv(input_csv)
    
    # Rename columns for clarity
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    
    if add_noise:
        # Add random noise to numerical columns
        print("Adding noise to the data...")
        numerical_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for col in numerical_cols:
            noise = np.random.normal(0, 0.5, len(df))
            df[col] = df[col] + noise
    
    if remove_samples:
        # Remove 20% of samples randomly
        print("Removing 20% of samples...")
        df = df.sample(frac=0.8, random_state=42)
    
    # Print dataset info
    print("\nDataset Summary:")
    print(f"Number of samples: {len(df)}")
    print("\nValue ranges:")
    for col in df.columns:
        if col != 'species':
            print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")
    
    # Save processed data
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw CSV file")
    parser.add_argument("--output", required=True, help="Path to save the processed CSV file")
    parser.add_argument("--add-noise", action="store_true", help="Add random noise to features")
    parser.add_argument("--remove-samples", action="store_true", help="Remove 20% of samples")
    args = parser.parse_args()

    preprocess(args.input, args.output, args.add_noise, args.remove_samples)