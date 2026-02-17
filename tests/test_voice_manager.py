import pytest
from bot_social_network.voice_manager import select_voice, _voice_cache


@pytest.fixture(autouse=True)
def mock_voice_cache():
    """Fixture to ensure the voice cache is in a known state for each test."""
    _voice_cache["male"] = ["en-US-Male-1", "en-US-Male-2"]
    _voice_cache["female"] = ["en-US-Female-1", "en-US-Female-2"]
    yield
    # Teardown - clear the cache after the test
    _voice_cache["male"] = []
    _voice_cache["female"] = []


def test_select_voice_consistency():
    """Tests that the same bot name always gets the same voice."""
    bot_name = "ConsistentBot"
    voice1 = select_voice(bot_name)
    voice2 = select_voice(bot_name)
    assert voice1 is not None
    assert voice1 == voice2


def test_select_voice_distribution():
    """Tests that different bot names can get different voices."""
    # This test is slightly more complex to guarantee different voices are selected
    # without relying on the internal implementation of hash() too much.

    all_voices = _voice_cache["male"] + _voice_cache["female"]
    if len(all_voices) < 2:
        pytest.skip("Not enough voices to test distribution.")

    found_voices = set()
    # Iterate through bot names until we find two that produce different voices
    for i in range(200):  # Limit to 200 attempts to prevent infinite loops
        bot_name = f"Bot_{i}"
        voice = select_voice(bot_name)
        if voice:
            found_voices.add(voice)
        if len(found_voices) > 1:
            break

    assert len(found_voices) > 1, (
        "Could not find two bot names that produce different voices."
    )


def test_select_voice_genders():
    """Tests that the hashing correctly assigns to male and female lists."""
    # Find a name that hashes to an even number for female voice
    female_bot_name = ""
    for i in range(100):
        name = f"Bot{i}"
        if hash(name) % 2 == 0:
            female_bot_name = name
            break

    # Find a name that hashes to an odd number for male voice
    male_bot_name = ""
    for i in range(100):
        name = f"Bot{i}"
        if hash(name) % 2 != 0:
            male_bot_name = name
            break

    assert female_bot_name, "Could not find a name that hashes to an even number"
    assert male_bot_name, "Could not find a name that hashes to an odd number"

    female_voice = select_voice(female_bot_name)
    male_voice = select_voice(male_bot_name)

    assert female_voice in _voice_cache["female"]
    assert male_voice in _voice_cache["male"]
