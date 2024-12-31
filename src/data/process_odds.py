import pandas as pd
import os
from src.utils.config import Config

# Dictionary to map divisions (to be populated)
division_mapping = {
    'E0': 'English Premier League',
    'D1': 'German Bundesliga',
    'SP1': 'Spanish La Liga',
    'F1': 'French Ligue 1',
    'I1': 'Italian Serie A',
}

def combine_odds_data(config: Config):
    # Get all CSV files in the odds directory
    odds_files = os.listdir(config.file_config.odds_path)
    
    # Initialize empty list to store dataframes
    dfs = []
    
    # Read and process each CSV file
    for file in odds_files:
        print(file)
        df = pd.read_csv(f'{config.file_config.odds_path}/{file}')
        
        # Map division codes if present in the dataframe
        if 'div' in df.columns:
            df['div'] = df['div'].map(division_mapping).fillna(df['div'])
            
        dfs.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_df = combined_df.rename(columns={'Date': 'date', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})

    combined_df['date'] = pd.to_datetime(combined_df['date'], format='mixed', dayfirst=True)

    return combined_df
