import pandas as pd
from src.utils.config import Config

def season_based_split(df, season_column, config):
    """
    Split the dataset into train, validation, and test sets based on season.
    """
    
    train_df = df[(df[season_column] != config.data_config.test_season) & (df[season_column] != config.data_config.eval_season)]
    test_df = df[df[season_column] == config.data_config.test_season]
    # eval_df = df[df[season_column] == config.data_config.eval_season]
    
    return train_df, test_df

if __name__ == '__main__':
    config = Config()

    df = pd.read_csv(config.file_config.input_features_path)
    train_df, test_df = season_based_split(df, 'season', config)
    
    train_df.to_csv(config.file_config.train_set_path, index=False)
    test_df.to_csv(config.file_config.test_set_path, index=False)
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")