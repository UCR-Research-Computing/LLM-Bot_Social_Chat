from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Bot(Base):
    __tablename__ = 'bots'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    persona = Column(Text, nullable=False)
    model = Column(String, default='gemini-1.5-flash')
    posts = relationship("Post", back_populates="bot", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="bot", cascade="all, delete-orphan")

class Post(Base):
    __tablename__ = 'posts'
    id = Column(Integer, primary_key=True)
    content = Column(Text)
    bot_id = Column(Integer, ForeignKey('bots.id'), nullable=True)
    bot = relationship("Bot", back_populates="posts")
    sender = Column(String)

class Memory(Base):
    __tablename__ = 'memories'
    id = Column(Integer, primary_key=True)
    key = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    bot_id = Column(Integer, ForeignKey('bots.id'), nullable=False)
    bot = relationship("Bot", back_populates="memories")

engine = create_engine('sqlite:///bots.db')
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
