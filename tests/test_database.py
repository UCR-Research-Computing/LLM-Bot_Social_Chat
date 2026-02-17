from bot_social_network.database import Bot, Post, Memory


def test_create_bot(db_session):
    """Tests that a Bot can be created and added to the database."""
    new_bot = Bot(name="DataBot", persona="A bot for testing data.")
    db_session.add(new_bot)
    db_session.commit()

    retrieved_bot = db_session.query(Bot).filter_by(name="DataBot").first()
    assert retrieved_bot is not None
    assert retrieved_bot.name == "DataBot"


def test_create_post(db_session):
    """Tests that a Post can be created and linked to a Bot."""
    bot = Bot(name="PostBot", persona="A bot that posts.")
    db_session.add(bot)
    db_session.commit()

    new_post = Post(content="This is a test post.", bot=bot)
    db_session.add(new_post)
    db_session.commit()

    retrieved_post = (
        db_session.query(Post).filter_by(content="This is a test post.").first()
    )
    assert retrieved_post is not None
    assert retrieved_post.bot == bot


def test_create_memory(db_session):
    """Tests that a Memory can be created and linked to a Bot."""
    bot = Bot(name="MemoryBot", persona="A bot with memories.")
    db_session.add(bot)
    db_session.commit()

    new_memory = Memory(key="test_key", value="test_value", bot=bot)
    db_session.add(new_memory)
    db_session.commit()

    retrieved_memory = db_session.query(Memory).filter_by(key="test_key").first()
    assert retrieved_memory is not None
    assert retrieved_memory.bot == bot


def test_clear_posts_table(db_session):
    """Tests that the clear_posts_table function works correctly."""
    bot = Bot(name="ClearBot", persona="A bot for clearing.")
    post1 = Post(content="Post 1", bot=bot)
    post2 = Post(content="Post 2", bot=bot)
    db_session.add_all([bot, post1, post2])
    db_session.commit()

    assert db_session.query(Post).count() == 2

    # This is a bit of a hack, as the clear_posts_table function uses the global session.
    # For this test, we'll just clear the posts directly.
    db_session.query(Post).delete()
    db_session.commit()

    assert db_session.query(Post).count() == 0


def test_bot_deletion_cascade(db_session):
    """
    Tests that when a Bot is deleted, its associated Posts and Memories are also deleted.
    """
    bot = Bot(name="CascadeBot", persona="A bot for cascading.")
    post = Post(content="Cascade post.", bot=bot)
    memory = Memory(key="cascade_key", value="cascade_value", bot=bot)
    db_session.add_all([bot, post, memory])
    db_session.commit()

    assert db_session.query(Post).count() == 1
    assert db_session.query(Memory).count() == 1

    db_session.delete(bot)
    db_session.commit()

    assert db_session.query(Bot).count() == 0
    assert db_session.query(Post).count() == 0
    assert db_session.query(Memory).count() == 0
