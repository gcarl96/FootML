import os
import json
import datetime
import pandas as pd
from src.data.build_features import build_features
from src.data.split_data import season_based_split
from src.models.train_model import train_model
from src.evaluation.evaluate_model import evaluate_model
from src.utils.config import Config

def run_experiment(config: Config):
    # Build features
    features = build_features(config)
 
    # Split data
    train_df, test_df = season_based_split(features, 'season', config)

    # Train model
    model, test_predictions = train_model(train_df, test_df, config)

    # Set up experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = f"experiments/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Evaluate model
    evaluation_results = evaluate_model(test_df, test_predictions, config, experiment_dir)

    test_df.loc[:, 'Pred_Prob'] = test_predictions[:, 1]
    test_df.to_csv(os.path.join(experiment_dir, 'predictions.csv'), index=False)

    # Save model and results
    results = {
        "model_parameters": config.model_config.__dict__,
        "data_parameters": config.data_config.__dict__,
        "evaluation_results": evaluation_results,
        "timestamp": timestamp
    }

    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    config = Config()
    run_experiment(config)

