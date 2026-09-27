"""SQLite storage: bots, posts, memories.

One engine per process, WAL mode, and a short-lived session per operation, so worker
threads never share a Session (the old module-level global Session was not
thread-safe and every to_thread call used it).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from sqlalchemy import (
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from . import settings


class Base(DeclarativeBase):
    pass


class Bot(Base):
    __tablename__ = "bots"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(255), default=settings.DEFAULT_MODEL)
    voice: Mapped[str | None] = mapped_column(String(64), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float, nullable=True)
    posts: Mapped[list["Post"]] = relationship(
        "Post", back_populates="bot", cascade="all, delete-orphan"
    )
    memories: Mapped[list["Memory"]] = relationship(
        "Memory", back_populates="bot", cascade="all, delete-orphan"
    )


class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    bot_id: Mapped[int | None] = mapped_column(ForeignKey("bots.id"), nullable=True)
    bot: Mapped["Bot | None"] = relationship("Bot", back_populates="posts")
    sender: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=datetime.now
    )
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), nullable=False)
    bot: Mapped["Bot"] = relationship("Bot", back_populates="memories")
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, default=datetime.now
    )


# Columns added after v0.1; an older bots.db gets them via ALTER TABLE.
_ADDED_COLUMNS = {
    "bots": {"voice": "VARCHAR(64)", "temperature": "FLOAT"},
    "posts": {
        "created_at": "DATETIME",
        "model": "VARCHAR(255)",
        "tokens_in": "INTEGER",
        "tokens_out": "INTEGER",
        "cost_usd": "FLOAT",
        "latency_ms": "INTEGER",
        "error": "TEXT",
    },
    "memories": {"created_at": "DATETIME"},
}


def _migrate(engine: Engine) -> None:
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in _ADDED_COLUMNS.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in cols.items():
                if name not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))


class Database:
    def __init__(self, url: str | None = None):
        if url is None:
            settings.ensure_dirs()
            url = f"sqlite:///{settings.DB_PATH}"
        self.url = url
        extra: dict[str, Any] = {}
        if ":memory:" in url:
            # One shared connection, or each thread would see an empty database.
            extra["poolclass"] = StaticPool
        self.engine = create_engine(
            url, connect_args={"check_same_thread": False, "timeout": 15}, **extra
        )
        if url.startswith("sqlite") and ":memory:" not in url:

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn: Any, _rec: Any) -> None:
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

        Base.metadata.create_all(self.engine)
        _migrate(self.engine)
        self._factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._factory()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # ---- reads -------------------------------------------------------------
    def bots(self) -> list[Bot]:
        with self.session() as s:
            return list(s.query(Bot).order_by(Bot.id).all())

    def bot(self, name: str) -> Bot | None:
        with self.session() as s:
            return s.query(Bot).filter_by(name=name).first()

    def recent_posts(self, limit: int = 50) -> list[Post]:
        """Newest first."""
        with self.session() as s:
            return list(s.query(Post).order_by(Post.id.desc()).limit(limit).all())

    def memories(self, bot_id: int, limit: int = 30) -> list[Memory]:
        """The newest `limit` memories for a bot, oldest first."""
        with self.session() as s:
            rows = (
                s.query(Memory)
                .filter_by(bot_id=bot_id)
                .order_by(Memory.id.desc())
                .limit(limit)
                .all()
            )
            return list(reversed(rows))

    def run_totals(self) -> dict[str, float | int]:
        with self.session() as s:
            row = s.execute(
                text(
                    "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), "
                    "COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0) FROM posts"
                )
            ).one()
            return {
                "posts": int(row[0]),
                "cost_usd": float(row[1]),
                "tokens_in": int(row[2]),
                "tokens_out": int(row[3]),
            }

    # ---- writes ------------------------------------------------------------
    def add_post(self, **fields: Any) -> Post:
        with self.session() as s:
            p = Post(**fields)
            s.add(p)
            s.flush()
            s.refresh(p)
            return p

    def add_memory(self, bot_id: int, key: str, value: str) -> Memory:
        with self.session() as s:
            m = Memory(bot_id=bot_id, key=key[:255], value=value)
            s.add(m)
            s.flush()
            return m

    def create_bot(
        self,
        name: str,
        persona: str,
        model: str,
        voice: str | None = None,
        temperature: float | None = None,
    ) -> Bot:
        with self.session() as s:
            b = Bot(
                name=name,
                persona=persona,
                model=model,
                voice=voice,
                temperature=temperature,
            )
            s.add(b)
            s.flush()
            return b

    def update_bot(self, bot_id: int, **fields: Any) -> None:
        with self.session() as s:
            b = s.get(Bot, bot_id)
            if b:
                for k, v in fields.items():
                    setattr(b, k, v)

    def delete_bot(self, bot_id: int) -> None:
        with self.session() as s:
            b = s.get(Bot, bot_id)
            if b:
                s.delete(b)

    def clear_posts(self) -> None:
        with self.session() as s:
            s.query(Post).delete()

    def replace_team(self, bots: list[dict[str, Any]]) -> None:
        """Replace every bot, memory and post with the given team in one transaction."""
        with self.session() as s:
            s.query(Post).delete()
            s.query(Memory).delete()
            s.query(Bot).delete()
            for data in bots:
                b = Bot(
                    name=data["name"],
                    persona=data["persona"],
                    model=data["model"],
                    voice=data.get("voice"),
                    temperature=data.get("temperature"),
                )
                s.add(b)
                for m in data.get("memories") or []:
                    s.add(Memory(key=m["key"], value=m["value"], bot=b))

    def export_team(self) -> list[dict[str, Any]]:
        with self.session() as s:
            out = []
            for b in s.query(Bot).order_by(Bot.id).all():
                d: dict[str, Any] = {
                    "name": b.name,
                    "persona": b.persona,
                    "model": b.model,
                }
                if b.voice:
                    d["voice"] = b.voice
                if b.temperature is not None:
                    d["temperature"] = b.temperature
                mems = [{"key": m.key, "value": m.value} for m in b.memories]
                if mems:
                    d["memories"] = mems
                out.append(d)
            return out

    def close(self) -> None:
        self.engine.dispose()
