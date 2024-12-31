import pandas as pd
import sqlite3
from src.utils.config import Config

match_data_query = """
    WITH team_gameweeks AS (
        -- Assign gameweek numbers to each team's matches
        SELECT 
            match_id,
            team,
            date,
            ROW_NUMBER() OVER (PARTITION BY team ORDER BY date) as gameweek
        FROM (
            -- Get all team matches as home and away
            SELECT 
                match_id,
                home_team as team,
                date
            FROM Match
            UNION ALL
            SELECT 
                match_id,
                away_team as team,
                date
            FROM Match
        ) 
    ),

    match_data AS (
        SELECT 
            m.match_id,
            m.date,
            m.season,
            m.competition,
            m.home_team,
            m.away_team,
            htg.gameweek as home_gameweek,
            atg.gameweek as away_gameweek,
            m.home_xG,
            m.away_xG,
            m.home_goals,
            m.away_goals,
            pi.player_id,
            pi.name,
            pi.home_away
        FROM Match m
        INNER JOIN team_gameweeks htg on m.match_id = htg.match_id and m.home_team = htg.team
        INNER JOIN team_gameweeks atg on m.match_id = atg.match_id and m.away_team = atg.team
        INNER JOIN Player_Info pi on m.match_id = pi.match_id
        where pi.started_match = 1
        and m.competition='Premier_League'
    )

    SELECT * FROM match_data
"""

def get_result(home_score, away_score):
    if home_score > away_score:
        return 1
    else:
        return 0

def get_match_data(config: Config):
    conn = sqlite3.connect(config.file_config.db_path)
    match_data = pd.read_sql_query(match_data_query, conn)

    match_data  = match_data.drop(columns=["name", "player_id", "home_away"]).drop_duplicates()
    match_data['result'] = match_data[['home_goals', 'away_goals']].apply(lambda x: get_result(x['home_goals'], x['away_goals']), axis=1)

    return match_data

if __name__ == '__main__':
    config = Config()
    match_data = get_match_data(config)
    print(match_data.head())