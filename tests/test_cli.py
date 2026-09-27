import pytest

from bot_social_network.cli import build_parser, main


def test_default_command_is_run():
    args = build_parser().parse_args([])
    assert args.cmd is None


def test_headless_flags_and_config_alias():
    a = build_parser().parse_args(
        [
            "headless",
            "--config",
            "fantasy_tavern",
            "--max-posts",
            "5",
            "--budget",
            "0.05",
        ]
    )
    assert a.team == "fantasy_tavern" and a.max_posts == 5 and a.budget == 0.05


def test_help_mentions_key_location(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "GEMINI_API_KEY" in out and "bot-social-network" in out


def test_headless_requires_a_stop_condition(capsys):
    assert main(["headless", "--team", "default"]) == 2
    assert "--max-posts or --duration" in capsys.readouterr().out


def test_teams_lists_bundled(capsys):
    assert main(["teams"]) == 0
    out = capsys.readouterr().out
    assert "fantasy_tavern" in out and "bundled" in out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "bot-social-network" in capsys.readouterr().out
