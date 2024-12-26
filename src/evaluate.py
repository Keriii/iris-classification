import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle
import argparse
import json
import yaml

def evaluate_model(data_csv, model_file, metrics_file):
    # Load data
    df = pd.read_csv(data_csv)
    X = df.drop("species", axis=1)
    y_true = df["species"]

    # Load model
    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Save detailed metrics in JSON
    metrics = {
        'accuracy': accuracy,
        'report': report
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save metrics in DVC-friendly YAML format
    dvc_metrics = {
        'accuracy': accuracy,
        'per_class': {
            class_name: {
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1-score': class_metrics['f1-score']
            }
            for class_name, class_metrics in report.items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        }
    }
    
    with open('metrics.yaml', 'w') as f:
        yaml.dump(dvc_metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the processed CSV")
    parser.add_argument("--model", required=True, help="Path to the trained model file")
    parser.add_argument("--metrics", required=True, help="Path to save metrics JSON")
    args = parser.parse_args()

    evaluate_model(args.input, args.model, args.metrics)