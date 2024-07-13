import sqlite3
import pandas as pd

class FootballDatabase:
    def __init__(self, db_path):
        self.db_path = db_path

    def connect(self):
        return sqlite3.connect(self.db_path)

    def get_player_match_stats(self):
        query = """
        SELECT player_id, match_id, shots, shots_on_target
        FROM Summary
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

    def get_match_results(self):
        query = """
        SELECT match_id, home_xG, away_xG
        FROM Match
        """
        with self.connect() as conn:
            return pd.read_sql_query(query, conn)

# Usage
# db = FootballDatabase('path/to/your/database.sqlite')
# player_stats = db.get_player_match_stats()
# match_results = db.get_match_results()