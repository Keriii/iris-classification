import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(data_path):
    # Read the data
    df = pd.read_csv(data_path)
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
    plt.title(f'Iris Dataset (Number of samples: {len(df)})')
    plt.savefig('data_visualization.png')
    plt.close()
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Number of samples: {len(df)}")
    print("\nFeature Statistics:")
    print(df.describe())

if __name__ == "__main__":
    visualize_data('data/preprocessed/iris_processed.csv')