from sqlalchemy import create_engine, ForeignKey, String, Text
from sqlalchemy.orm import relationship, sessionmaker, DeclarativeBase, Mapped, mapped_column
from typing import List

class Base(DeclarativeBase):
    pass


class Bot(Base):
    __tablename__ = "bots"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(255), default="gemini-1.5-flash")
    posts: Mapped[List["Post"]] = relationship("Post", back_populates="bot", cascade="all, delete-orphan")
    memories: Mapped[List["Memory"]] = relationship(
        "Memory", back_populates="bot", cascade="all, delete-orphan"
    )


class Post(Base):
    __tablename__ = "posts"
    id: Mapped[int] = mapped_column(primary_key=True)
    content: Mapped[str] = mapped_column(Text)
    bot_id: Mapped[int | None] = mapped_column(ForeignKey("bots.id"), nullable=True)
    bot: Mapped["Bot"] = relationship("Bot", back_populates="posts")
    sender: Mapped[str | None] = mapped_column(String(255), nullable=True)


class Memory(Base):
    __tablename__ = "memories"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), nullable=False)
    bot: Mapped["Bot"] = relationship("Bot", back_populates="memories")


engine = create_engine("sqlite:///bots.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


def clear_posts_table():
    """Clears all records from the posts table."""
    session.query(Post).delete()
    session.commit()


def close_database_connection():
    """Closes the session and disposes of the engine."""
    session.close()
    engine.dispose()


# --- Database Helper Functions ---


def db_add_memory(memory: Memory):
    session.add(memory)
    session.commit()


def db_create_bot(name, persona, model):
    new_bot = Bot(name=name, persona=persona, model=model)
    session.add(new_bot)
    session.commit()


def db_edit_bot(bot, name, persona, model):
    bot.name = name
    bot.persona = persona
    bot.model = model
    session.commit()


def db_delete_bot(bot):
    session.delete(bot)
    session.commit()


def db_clear_posts():
    session.query(Post).delete()
    session.commit()
