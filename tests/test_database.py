import sqlite3

from bot_social_network.database import Database, Post


def test_team_round_trip(db):
    db.replace_team(
        [
            {
                "name": "A",
                "persona": "p",
                "model": "gemini-3.8-flash",
                "voice": "Kore",
                "temperature": 0.7,
                "memories": [{"key": "k", "value": "v"}],
            },
            {"name": "B", "persona": "q", "model": "gemma4:e4b"},
        ]
    )
    team = db.export_team()
    assert [b["name"] for b in team] == ["A", "B"]
    assert team[0]["voice"] == "Kore" and team[0]["temperature"] == 0.7
    assert team[0]["memories"] == [{"key": "k", "value": "v"}]
    assert "memories" not in team[1]


def test_replace_team_clears_posts_and_memories(db):
    db.replace_team([{"name": "A", "persona": "p", "model": "m"}])
    a = db.bot("A")
    db.add_post(content="hi", sender="A", bot_id=a.id)
    db.add_memory(a.id, "k", "v")
    db.replace_team([{"name": "Z", "persona": "p", "model": "m"}])
    assert db.recent_posts() == []
    assert [b.name for b in db.bots()] == ["Z"]


def test_post_metrics_and_totals(db):
    db.add_post(content="x", sender="A", tokens_in=10, tokens_out=5, cost_usd=0.001)
    db.add_post(content="y", sender="SYSTEM", error="boom")
    t = db.run_totals()
    assert t["posts"] == 2 and t["tokens_in"] == 10
    assert abs(t["cost_usd"] - 0.001) < 1e-9
    newest = db.recent_posts(1)[0]
    assert isinstance(newest, Post) and newest.error == "boom"


def test_delete_bot_cascades(db):
    db.replace_team([{"name": "A", "persona": "p", "model": "m"}])
    a = db.bot("A")
    db.add_post(content="hi", sender="A", bot_id=a.id)
    db.add_memory(a.id, "k", "v")
    db.delete_bot(a.id)
    assert db.bots() == [] and db.recent_posts() == []


def test_memories_returns_newest_limit_oldest_first(db):
    db.replace_team([{"name": "A", "persona": "p", "model": "m"}])
    a = db.bot("A")
    for i in range(5):
        db.add_memory(a.id, f"k{i}", "v")
    assert [m.key for m in db.memories(a.id, limit=3)] == ["k2", "k3", "k4"]


def test_old_v01_database_is_migrated(tmp_path):
    """A v0.1 bots.db (no new columns) opens, keeps its data and gains the columns."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE bots (id INTEGER PRIMARY KEY, name VARCHAR(255) UNIQUE NOT NULL,
                           persona TEXT NOT NULL, model VARCHAR(255));
        CREATE TABLE posts (id INTEGER PRIMARY KEY, content TEXT, bot_id INTEGER,
                            sender VARCHAR(255));
        CREATE TABLE memories (id INTEGER PRIMARY KEY, key VARCHAR(255) NOT NULL,
                               value TEXT NOT NULL, bot_id INTEGER NOT NULL);
        INSERT INTO bots VALUES (1, 'Old', 'persona', 'gemini-1.5-flash');
        INSERT INTO posts VALUES (1, 'hello', 1, 'Old');
        """
    )
    con.close()
    d = Database(f"sqlite:///{path}")
    p = d.recent_posts()[0]
    assert p.content == "hello" and p.cost_usd is None
    assert d.bot("Old").voice is None
    d.close()


def test_file_database_uses_wal(tmp_path):
    d = Database(f"sqlite:///{tmp_path / 'x.db'}")
    d.bots()
    with d.engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
    d.close()
