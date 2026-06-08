# modelling untuk Workflow-CI
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'churn_preprocessing.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog()
    
    # Set experiment
    # mlflow.set_experiment("CI_Churn_Prediction")
    
    # Train model with MLflow tracking
    with mlflow.start_run(run_name=f"CI_RF_n{args.n_estimators}_d{args.max_depth}"):
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        
        # Train model
        print(f"Training RandomForest with n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
        
        model.fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        # Log metrics manually (extra)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        
        print("\n✅ Training completed successfully!")
        print(f"MLflow run_id: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    main()
