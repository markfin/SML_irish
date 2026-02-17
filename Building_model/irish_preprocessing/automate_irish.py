import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import os

def preprocess_iris_data():
    # Load the Iris dataset
    iris = load_iris()

    # Create a DataFrame
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['target_names'] = iris.target_names[iris.target]

    numerical_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # Separate features (X) and target (y)
    X = iris_df[numerical_features]
    y = iris_df['target']

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Apply scaler to numerical features
    X_scaled_array = scaler.fit_transform(X)

    # Convert scaled features back to DataFrame
    X_scaled = pd.DataFrame(X_scaled_array, columns=numerical_features)

    # Determine the script's directory
    script_dir = os.path.dirname(__file__)
    # Construct the path to the processed data directory relative to the script's location
    output_dir = os.path.join(script_dir, '..', 'data', 'processed')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed data
    X_scaled.to_csv(os.path.join(output_dir, 'X_processed.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'y_processed.csv'), index=False)

    print(f'Processed features saved to {os.path.join(output_dir, 'X_processed.csv')}')
    print(f'Processed target saved to {os.path.join(output_dir, 'y_processed.csv')}')

if __name__ == '__main__':
    preprocess_iris_data()

print('File automate_irish.py updated with correct output path successfully.')
