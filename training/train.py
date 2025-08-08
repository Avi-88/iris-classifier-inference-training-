import os
import sys
import argparse
import joblib
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main(args):
    """
    Main function to orchestrate the model training and validation process.
    """
    print("--- Starting Model Training Run ---")

    # Configure MLflow client to connect to the in-cluster service
    # This address is discoverable via Kubernetes DNS.
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("iris-classifier-production")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Log parameters passed from the workflow
        mlflow.log_param("git_commit_sha", os.getenv("GIT_COMMIT_SHA", "unknown"))

        # 1. Load Data
        iris = load_iris()
        X, y = iris.data, iris.target

        # 2. Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Dataset loaded. Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        mlflow.log_param("test_set_size", len(X_test))

        # 3. Train a simple Logistic Regression model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        print("Model training complete.")

        # 4. Validate the model on the test set
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy on test data: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)

        # 5. Perform validation check against a business-defined threshold
        ACCURACY_THRESHOLD = 0.90
        mlflow.log_param("accuracy_threshold", ACCURACY_THRESHOLD)

        if accuracy >= ACCURACY_THRESHOLD:
            print(f"Validation PASSED. Accuracy ({accuracy:.4f}) is at or above threshold ({ACCURACY_THRESHOLD}).")
            
            # Save the trained model to the path provided by the Argo workflow
            joblib.dump(model, args.model_output_path)
            print(f"Model artifact saved to: {args.model_output_path}")

            # Log the model to the MLflow registry. This is the action that
            # will trigger the webhook for the CD pipeline.
            mlflow.sklearn.log_model(model, "model")
            print("Model successfully logged to MLflow registry.")
            
            # Exit with success code for Argo Workflows
            sys.exit(0)
        else:
            print(f"Validation FAILED. Accuracy ({accuracy:.4f}) is below threshold ({ACCURACY_THRESHOLD}).")
            print("Model will not be saved or logged.")
            
            # Exit with a failure code. Argo Workflows will see this and mark the step as failed.
            sys.exit(1)

if __name__ == "__main__":
    # This script expects a command-line argument to know where to save the model file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-path", type=str, required=True, help="Path to save the trained model artifact.")
    args = parser.parse_args()
    main(args)