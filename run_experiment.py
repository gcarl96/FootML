import os
import datetime
import pandas as pd
from src.data.build_features import build_features
from src.data.split_data import time_based_split
from src.models.train_model import train_model, evaluate_model
from src.utils.config import Config

def run_experiment(config: Config):


    # Set up experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = f"experiments/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Build features
    features = build_features(config)

    # Split data
    train_df, val_df, test_df = time_based_split(features, 'date')

    # Train model
    model = train_model(train_df)

    # Evaluate model
    evaluation_results = evaluate_model(model, val_df)

    # Save model and results
    save_model(model, os.path.join(experiment_dir, 'model.pkl'))
    save_results(evaluation_results, os.path.join(experiment_dir, 'results.json'))

    print(f"Experiment completed and saved in {experiment_dir}")

if __name__ == "__main__":
    config = Config()
    run_experiment(config)

