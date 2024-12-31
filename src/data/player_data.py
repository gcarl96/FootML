import pandas as pd
import sqlite3
from src.utils.config import Config

player_data_query = """
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
    )

    SELECT 
        pi.match_id,
        pi.player_id,
        pi.name,
        m.date,
        m.season,
        m.competition,
        pi.home_away,
        CASE 
            WHEN pi.home_away = 'H' THEN m.home_team 
            ELSE m.away_team 
        END as team,
        tg.gameweek,
        CAST(pi.started_match AS INTEGER) as started_match,
        pi.minutes,
        COALESCE(s.goals, 0) as goals,
        COALESCE(s.assists, 0) as assists,
        COALESCE(s.shots, 0) as shots,
        COALESCE(s.shots_on_target, 0) as shots_on_target,
        COALESCE(s.xG, 0) as xG,
        COALESCE(s.xA, 0) as xA,
        COALESCE(s.passes_completed, 0) as passes_completed,
        COALESCE(s.passes_attempted, 0) as passes_attempted,
        COALESCE(s.progressive_passes, 0) as progressive_passes,
        COALESCE(s.carries, 0) as carries,
        COALESCE(s.progressive_carries, 0) as progressive_carries,
        COALESCE(s.successful_dribbles, 0) as successful_dribbles,
        COALESCE(s.dribbles_attempted, 0) as dribbles_attempted
    FROM Player_Info pi
    JOIN Match m ON pi.match_id = m.match_id
    JOIN team_gameweeks tg ON 
        tg.match_id = m.match_id AND
        tg.team = CASE 
            WHEN pi.home_away = 'H' THEN m.home_team 
            ELSE m.away_team 
        END
    LEFT JOIN Summary s ON 
        pi.match_id = s.match_id AND 
        pi.player_id = s.player_id
    ORDER BY team, gameweek DESC;
"""

def get_player_data(config: Config):
    conn = sqlite3.connect(config.file_config.db_path)
    player_data = pd.read_sql_query(player_data_query, conn)
    return player_data

if __name__ == '__main__':
    config = Config()
    player_data = get_player_data(config)
    print(player_data.head())