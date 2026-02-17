import pytest
from unittest.mock import patch
from bot_social_network.simulation import _parse_memory_string, get_available_models


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("key: value", ("key", "value")),
        ("  key  :  value  ", ("key", "value")),
        (
            "complex key: a value with a : colon",
            ("complex key", "a value with a : colon"),
        ),
        ("key:value", ("key", "value")),
        ("None", None),
        ("none", None),
        ("NONE", None),
        ("No significant memory was formed.", None),
        ("key value", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_memory_string(input_string, expected_output):
    """
    Tests the memory parsing logic with various valid and invalid inputs.
    """
    assert _parse_memory_string(input_string) == expected_output


def test_get_available_models_includes_all_gemini_models():
    """
    Tests that the static list of Gemini models is present, even if Ollama fails.
    """
    expected_gemini_models = [
        ("Gemini 1.5 Flash", "gemini-1.5-flash"),
        ("Gemini 1.5 Pro", "gemini-1.5-pro"),
        ("Gemini 2.5 Flash", "gemini-2.5-flash"),
        ("Gemini 2.5 Pro", "gemini-2.5-pro"),
    ]

    # Mock the subprocess.run to simulate Ollama not being installed
    with patch("subprocess.run", side_effect=FileNotFoundError):
        models = get_available_models()
        # Check that all expected Gemini models are in the returned list
        assert all(gemini_model in models for gemini_model in expected_gemini_models)
