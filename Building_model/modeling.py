
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_log_model():
    # Enable autologging for scikit-learn
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # Determine the script's directory
        script_dir = os.path.dirname(__file__)

        # Construct paths to preprocessed data relative to the script's location
        X_path = os.path.join(script_dir, '..', 'data', 'processed', 'X_processed.csv')
        y_path = os.path.join(script_dir, '..', 'data', 'processed', 'y_processed.csv')

        # Load preprocessed data
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).squeeze() # .squeeze() to convert DataFrame to Series if it's a single column

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Logistic Regression model
        model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Log custom metrics (autologging will handle most, but can explicitly log if needed)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Model trained with accuracy: {accuracy}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    # Set MLflow tracking URI to a local directory
    # Ensure 'mlruns' directory exists where the script is run or a custom path is set
    mlflow.set_tracking_uri("file:./mlruns") # Local MLflow tracking server
    train_and_log_model()
