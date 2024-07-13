import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_model(data_path):
    # Load data
    data = pd.read_csv(data_path)

    # Prepare features and target
    X = data[['shots', 'shots_on_target']]
    y = data['result']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a simple model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"Model Accuracy: {accuracy}")
        print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    mlflow.set_experiment("football_score_prediction")
    train_model('data/processed/football_data.csv')