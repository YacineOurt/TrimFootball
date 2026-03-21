import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "trimfootball.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            team_home TEXT NOT NULL,
            team_away TEXT NOT NULL,
            score_home INTEGER,
            score_away INTEGER,
            video_path TEXT NOT NULL,
            tracking_path TEXT NOT NULL DEFAULT '',
            fps REAL DEFAULT 0,
            total_frames INTEGER DEFAULT 0,
            half_frame INTEGER DEFAULT 0,
            team_home_id INTEGER DEFAULT 0,
            status TEXT DEFAULT 'pending',
            match_dir TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def create_match(**kwargs):
    conn = get_db()
    cursor = conn.execute(
        """INSERT INTO matches (date, team_home, team_away, score_home, score_away,
           video_path, tracking_path, fps, total_frames, half_frame, team_home_id,
           status, match_dir)
           VALUES (:date, :team_home, :team_away, :score_home, :score_away,
           :video_path, :tracking_path, :fps, :total_frames, :half_frame, :team_home_id,
           :status, :match_dir)""",
        kwargs,
    )
    conn.commit()
    match_id = cursor.lastrowid
    conn.close()
    return match_id


def update_match(match_id, **kwargs):
    conn = get_db()
    sets = ", ".join(f"{k} = :{k}" for k in kwargs)
    kwargs["id"] = match_id
    conn.execute(f"UPDATE matches SET {sets} WHERE id = :id", kwargs)
    conn.commit()
    conn.close()


def get_match(match_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_matches():
    conn = get_db()
    rows = conn.execute("SELECT * FROM matches ORDER BY date DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_match(match_id):
    conn = get_db()
    conn.execute("DELETE FROM matches WHERE id = ?", (match_id,))
    conn.commit()
    conn.close()


def get_matches_by_team(team_name):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM matches WHERE LOWER(team_home) = LOWER(?) OR LOWER(team_away) = LOWER(?) ORDER BY date DESC",
        (team_name, team_name),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_teams():
    conn = get_db()
    rows = conn.execute(
        "SELECT DISTINCT team FROM ("
        "  SELECT team_home AS team FROM matches"
        "  UNION"
        "  SELECT team_away FROM matches"
        ") ORDER BY team"
    ).fetchall()
    conn.close()
    return [r["team"] for r in rows]
