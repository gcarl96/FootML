from dataclasses import dataclass, field

@dataclass
class FileConfig:
    # Main filepaths for running full pipeline
    db_path: str = 'data/raw/master.db'
    odds_path: str = 'data/raw/odds/'
    processed_data_path: str = 'data/processed/'
    outputs_path: str = 'data/outputs/'

    # Individual filepaths for each step
    odds_output_path: str = 'data/processed/odds.csv'
    input_features_path: str = 'data/processed/input_features.csv'
    train_set_path: str = 'data/processed/train_set.csv'
    test_set_path: str = 'data/processed/test_set.csv'

@dataclass
class DataConfig:
    rolling_gameweeks: int = 10
    target_column: str = 'result'
    test_season: str = '2023-2024'
    eval_season: str = '2024-2025'
    feature_list = [
        'home_rolling_goals', 'away_rolling_goals',
        'home_rolling_xg', 'away_rolling_xg',
        'goals_prior_home', 'goals_prior_away',
        'assists_prior_home', 'assists_prior_away', 
        'xG_prior_home', 'xG_prior_away',
        'xA_prior_home', 'xA_prior_away',
        'passes_completed_prior_home', 'passes_completed_prior_away',
        'passes_attempted_prior_home', 'passes_attempted_prior_away',
        'progressive_passes_prior_home', 'progressive_passes_prior_away',
        'carries_prior_home', 'carries_prior_away',
        'progressive_carries_prior_home', 'progressive_carries_prior_away',
        'successful_dribbles_prior_home', 'successful_dribbles_prior_away',
        'dribbles_attempted_prior_home', 'dribbles_attempted_prior_away',
        'appearances_home', 'appearances_away'
    ]


@dataclass
class ModelConfig:
    model_name: str = 'logistic_regression'
    # XGBoost
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    random_state: int = 42
    # Logistic Regression
    random_state: int = 42


@dataclass
class Config:
    file_config: FileConfig = field(default_factory=FileConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
