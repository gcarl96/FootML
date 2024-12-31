import pandas as pd
from src.utils.config import Config
from src.data.player_data import get_player_data
from src.data.match_data import get_match_data
from src.data.process_odds import combine_odds_data
def rolling_player_stats(player_stats: pd.DataFrame, config: Config):
    # Create a copy of player stats to join with itself
    rolling_player_stats = player_stats.merge(
        player_stats[['match_id', 'player_id', 'gameweek', 'home_away', 'goals', 'assists', 'xG', 'xA', 
                    'passes_completed', 'passes_attempted', 'progressive_passes',
                    'carries', 'progressive_carries', 'successful_dribbles', 'dribbles_attempted']],
        on='player_id',
        how='left',
        suffixes=('', '_prior')
    )

    # Filter to only include prior games within 10 gameweeks
    rolling_player_stats = rolling_player_stats[
        (rolling_player_stats['gameweek_prior'] < rolling_player_stats['gameweek']) & 
        (rolling_player_stats['gameweek'] - rolling_player_stats['gameweek_prior'] <= config.feature_config.rolling_gameweeks)
    ]

    # Calculate rolling averages for each player's stats
    rolling_player_stats = rolling_player_stats.groupby(['match_id', 'player_id', 'gameweek', 'home_away']).agg({
        'goals_prior': 'mean',
        'assists_prior': 'mean', 
        'xG_prior': 'mean',
        'xA_prior': 'mean',
        'passes_completed_prior': 'mean',
        'passes_attempted_prior': 'mean',
        'progressive_passes_prior': 'mean',
        'carries_prior': 'mean',
        'progressive_carries_prior': 'mean',
        'successful_dribbles_prior': 'mean',
        'dribbles_attempted_prior': 'mean',
        'match_id_prior': 'count'  # Count number of prior appearances
    }).rename(columns={'match_id_prior': 'appearances'}).reset_index()

    return rolling_player_stats

def rolling_team_stats(match_data: pd.DataFrame, config: Config):
    # Create team records by combining home and away stats
    team_gameweek_stats = pd.concat([
        match_data[['home_team', 'home_gameweek', 'home_xG', 'home_goals']].rename(
            columns={
                'home_team': 'team_name',
                'home_gameweek': 'gameweek',
                'home_xG': 'xg',
                'home_goals': 'goals'
            }
        ),
        match_data[['away_team', 'away_gameweek', 'away_xG', 'away_goals']].rename(
            columns={
                'away_team': 'team_name', 
                'away_gameweek': 'gameweek',
                'away_xG': 'xg',
                'away_goals': 'goals'
            }
        )
    ])

    # Sort by team and gameweek, remove duplicates
    team_gameweek_stats = (team_gameweek_stats
        .sort_values(['team_name', 'gameweek'], ascending=[True, False])
        .drop_duplicates()
        .reset_index()
    )

    # Calculate rolling averages for goals and xG
    for stat in ['goals', 'xg']:
        team_gameweek_stats[f'rolling_{stat}'] = (team_gameweek_stats
            .sort_values('gameweek')
            .groupby('team_name')[stat]
            .transform(lambda x: x.rolling(window=config.feature_config.rolling_gameweeks, min_periods=1).mean().shift(1))
        )

    match_data_with_priors = pd.merge(match_data, team_gameweek_stats[['team_name', 'gameweek', 'rolling_goals', 'rolling_xg']]
        .rename(columns={'rolling_goals' : 'home_rolling_goals', 'rolling_xg' : 'home_rolling_xg'}), left_on=['home_team', 'home_gameweek'], right_on=['team_name', 'gameweek']).drop(columns=['team_name', 'gameweek'])

    match_data_with_priors = pd.merge(match_data_with_priors, team_gameweek_stats[['team_name', 'gameweek', 'rolling_goals', 'rolling_xg']]
        .rename(columns={'rolling_goals' : 'away_rolling_goals', 'rolling_xg' : 'away_rolling_xg'}), left_on=['away_team', 'away_gameweek'], right_on=['team_name', 'gameweek']).drop(columns=['team_name', 'gameweek'])

    return match_data_with_priors

def merge_features(team_stats: pd.DataFrame, player_stats: pd.DataFrame, config: Config):
    # Merge match events with rolling player stats
    match_player_stats = pd.merge(
        team_stats,
        player_stats,
        on=['match_id']
    )

    # print(match_player_stats.columns)

    # Calculate average stats per match for home and away teams
    match_agg_stats = match_player_stats.groupby(['match_id', 'home_away']).agg({
        'goals_prior': 'mean',
        'assists_prior': 'mean',
        'xG_prior': 'mean', 
        'xA_prior': 'mean',
        'passes_completed_prior': 'mean',
        'passes_attempted_prior': 'mean',
        'progressive_passes_prior': 'mean',
        'carries_prior': 'mean',
        'progressive_carries_prior': 'mean',
        'successful_dribbles_prior': 'mean',
        'dribbles_attempted_prior': 'mean',
        'appearances': 'mean'
    }).reset_index()

    # Pivot home/away stats into separate columns
    match_stats_pivoted = match_agg_stats.pivot(
        index='match_id',
        columns='home_away',
        values=['goals_prior', 'assists_prior', 'xG_prior', 'xA_prior',
                'passes_completed_prior', 'passes_attempted_prior', 'progressive_passes_prior',
                'carries_prior', 'progressive_carries_prior', 'successful_dribbles_prior',
                'dribbles_attempted_prior', 'appearances']
    ).reset_index()

    # Flatten column names
    match_stats_pivoted.columns = [
        f'{col[0]}_{"home" if col[1]=="H" else "away"}' if col[1] in ['H','A'] else 'match_id' 
        for col in match_stats_pivoted.columns
    ]

    # Merge back with original match events
    final_match_stats = pd.merge(
        team_stats,
        match_stats_pivoted,
        on='match_id'
    )

    return final_match_stats

def merge_odds(input_stats: pd.DataFrame, config: Config):
    odds_data = combine_odds_data(config)

    input_stats['date'] = pd.to_datetime(input_stats['date'], format='mixed', dayfirst=True)
    input_stats = pd.merge(input_stats, odds_data[['date', 'home_team', 'away_team', 'B365H', 'B365D', 'B365A']], left_on=['date', 'home_team', 'away_team'], right_on=['date', 'home_team', 'away_team'], how='left')

    return input_stats


def build_features(config: Config):
    player_data = get_player_data(config)
    match_data = get_match_data(config)

    team_stats = rolling_team_stats(match_data, config)
    player_stats = rolling_player_stats(player_data, config)

    merged_features = merge_features(team_stats, player_stats, config)

    features_with_odds = merge_odds(merged_features, config)
    
    return features_with_odds

if __name__ == '__main__':
    config = Config()
    features = build_features(config)
        
    print(features[(features["home_team"] == "Arsenal") | (features["away_team"] == "Arsenal")].sort_values(by='date', ascending=False))