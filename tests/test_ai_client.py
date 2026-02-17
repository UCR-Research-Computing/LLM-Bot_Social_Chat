import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from bot_social_network.database import Bot, Post, Memory
from bot_social_network.ai_client import (
    _build_prompt,
    _build_memory_prompt,
    generate_post_gemini,
    generate_post_ollama,
)


@pytest.fixture
def sample_bot():
    """Provides a sample Bot object for testing."""
    return Bot(name="TestBot", persona="A test persona.", model="test_model")


# --- Test Core Prompt Building ---


def test_build_prompt_no_history_no_memories(sample_bot):
    """
    Tests that the prompt is correctly generated for a bot with no memories and no post history.
    """
    prompt = _build_prompt(sample_bot, [], [], [])
    assert "You are an AI named TestBot." in prompt
    assert "Your persona is: 'A test persona.'." in prompt
    assert "Here are some of your core memories and beliefs:" not in prompt
    assert "Here are the recent posts in the conversation:" not in prompt
    assert "What is on your mind?" in prompt


def test_build_prompt_with_memories(sample_bot):
    """
    Tests that the prompt correctly includes the bot's memories.
    """
    memories = [
        Memory(key="favorite_color", value="blue"),
        Memory(key="mission", value="To boldly go where no bot has gone before."),
    ]
    prompt = _build_prompt(sample_bot, [], [], memories)
    assert "Here are some of your core memories and beliefs:" in prompt
    assert "- favorite_color: blue" in prompt
    assert "- mission: To boldly go where no bot has gone before." in prompt


def test_build_prompt_with_history(sample_bot):
    """
    Tests that the prompt correctly includes the recent post history.
    """
    posts = [
        Post(sender="Alice", content="Hello, world!"),
        Post(sender="Bob", content="This is a test."),
    ]
    prompt = _build_prompt(sample_bot, ["Alice", "Bob"], posts, [])
    assert "Here are the recent posts in the conversation:" in prompt
    assert "- @Alice: Hello, world!" in prompt
    assert "- @Bob: This is a test." in prompt


def test_build_prompt_with_everything(sample_bot):
    """
    Tests that the prompt is correctly generated with memories, history, and other bots.
    """
    memories = [Memory(key="home_planet", value="Cybertron")]
    posts = [Post(sender="Alice", content="First post!")]
    other_bots = ["Alice", "Charlie"]

    prompt = _build_prompt(sample_bot, other_bots, posts, memories)

    assert "You are in a conversation with: @Alice, @Charlie." in prompt
    assert "Here are some of your core memories and beliefs:" in prompt
    assert "- home_planet: Cybertron" in prompt
    assert "Here are the recent posts in the conversation:" in prompt
    assert "- @Alice: First post!" in prompt
    assert (
        "Based on these posts and your memories, what is your thoughtful reaction?"
        in prompt
    )


def test_build_memory_prompt(sample_bot):
    """
    Tests that the memory generation prompt is correctly constructed.
    """
    posts = [
        Post(sender="Alice", content="What is the meaning of life?"),
        Post(sender="TestBot", content="42, obviously."),
    ]
    prompt = _build_memory_prompt(sample_bot, posts)
    assert "You are an AI named TestBot." in prompt
    assert "You have just participated in a conversation." in prompt
    assert "- @TestBot: 42, obviously." in prompt
    assert "- @Alice: What is the meaning of life?" in prompt
    assert "Generate a new memory in the format 'key: value'." in prompt


# --- Test AI Client Functions ---


@patch("bot_social_network.ai_client.genai.GenerativeModel")
def test_generate_post_gemini(mock_genai_model, sample_bot):
    """
    Tests the Gemini API call, mocking the genai library.
    """
    # Arrange: Set up the mock to return a predictable response
    mock_response = MagicMock()
    mock_response.text = "This is a test response from Gemini."

    mock_model_instance = MagicMock()
    mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
    mock_genai_model.return_value = mock_model_instance

    # Act: Call the async function using asyncio.run
    response_text, prompt = asyncio.run(generate_post_gemini(sample_bot, [], [], []))

    # Assert: Check that the function returned the expected text
    assert response_text == "This is a test response from Gemini."
    assert "You are an AI named TestBot" in prompt
    mock_genai_model.assert_called_with(sample_bot.model)
    mock_model_instance.generate_content_async.assert_called_once()


@patch("bot_social_network.ai_client.subprocess.run")
def test_generate_post_ollama(mock_subprocess_run, sample_bot):
    """
    Tests the Ollama system call, mocking the subprocess.run call.
    """
    # Arrange: Set up the mock to return a predictable response
    mock_process_result = MagicMock()
    mock_process_result.stdout = "This is a test response from Ollama."
    mock_process_result.stderr = ""
    mock_process_result.returncode = 0
    mock_subprocess_run.return_value = mock_process_result

    # Act: Call the async function using asyncio.run
    response_text, prompt = asyncio.run(generate_post_ollama(sample_bot, [], [], []))

    # Assert: Check that the function returned the expected text
    assert response_text == "This is a test response from Ollama."
    assert "You are an AI named TestBot" in prompt
    # Note: ai_client._run_ollama_sync calls subprocess.run
    # Since we mock subprocess.run, it should be called.
    # However, since generate_post_ollama runs in a thread pool,
    # mocking subprocess.run at the module level might be tricky if imports are already bound.
    # But since we patched 'bot_social_network.ai_client.subprocess.run', it should work.
    mock_subprocess_run.assert_called_once()
