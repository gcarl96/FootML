from dataclasses import dataclass, field

@dataclass
class FileConfig:
    db_path: str = 'data/raw/master.db'
    odds_path: str = 'data/raw/odds/'
    processed_data_path: str = 'data/processed/'
    outputs_path: str = 'data/outputs/'


@dataclass
class FeatureConfig:
    rolling_gameweeks: int = 10


@dataclass
class ModelConfig:
    model_name: str = 'xgboost'
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    random_state: int = 42


@dataclass
class Config:
    file_config: FileConfig = field(default_factory=FileConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
