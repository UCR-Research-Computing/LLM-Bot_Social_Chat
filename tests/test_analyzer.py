import os
import json
from bot_social_network.analyzer import analyze_log


def test_analyze_log_generates_report(tmp_path):
    # Create a mock simulation.jsonl
    log_file = tmp_path / "simulation.jsonl"
    logs = [
        {"asctime": "2026-02-16 22:45:31,946", "event": "system.init"},
        {
            "asctime": "2026-02-16 22:45:37,218",
            "event": "post.generated",
            "bot_name": "Dan",
            "post_content": "Hello @Steve",
            "bot_model": "gemini-2.5-flash",
        },
        {
            "asctime": "2026-02-16 22:45:42,969",
            "event": "post.generated",
            "bot_name": "Steve",
            "post_content": "Hi @Dan, how are you?",
            "bot_model": "gemini-2.5-flash",
        },
        {"asctime": "2026-02-16 22:45:51,219", "event": "sim.end.max_posts"},
    ]

    with open(log_file, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + chr(10))

    # Run analyzer
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        analyze_log(str(log_file))

        assert os.path.exists("analysis_report.html")
        with open("analysis_report.html", "r") as f:
            content = f.read()
            assert "Simulation Analysis Report" in content
            assert "Dan" in content
            assert "Steve" in content
    finally:
        os.chdir(original_cwd)


def test_analyze_log_no_posts(tmp_path, capsys):
    log_file = tmp_path / "empty.jsonl"
    with open(log_file, "w") as f:
        f.write(json.dumps({"event": "system.init"}) + chr(10))

    analyze_log(str(log_file))
    captured = capsys.readouterr()
    assert "No posts found" in captured.out
