from src.data.database import FootballDatabase

def test_database_connection():
    db = FootballDatabase('data/raw/master.db')
    assert db.connect() is not None

def test_get_player_match_stats():
    db = FootballDatabase('data/raw/football_data.sqlite')
    stats = db.get_player_match_stats()
    assert not stats.empty