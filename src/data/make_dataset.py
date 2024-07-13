from database import FootballDatabase
import pandas as pd

def create_dataset(db_path, output_path):
    db = FootballDatabase(db_path)
    player_stats = db.get_player_match_stats()
    match_results = db.get_match_results()

    # Aggregate player stats for each team in each match
    team_stats = player_stats.groupby('match_id').agg({
        'shots': 'mean',
        'shots_on_target': 'mean',
    }).reset_index()

    # Merge with match results
    full_dataset = pd.merge(team_stats, match_results, on='match_id')

    # Create target variable (e.g., home team win/loss/draw)
    full_dataset['result'] = (full_dataset['home_xG'] > full_dataset['away_xG']).astype(int)

    # Save to CSV
    full_dataset.to_csv(output_path, index=False)
    return full_dataset

if __name__ == '__main__':
    dataset = create_dataset('data/raw/master.db', 'data/processed/football_data.csv')
    print(dataset.head())
    print(dataset.info())