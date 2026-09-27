import json

from bot_social_network import settings
from bot_social_network.analyzer import analyze_cli, analyze_log, resolve_log

LOGS = [
    {"asctime": "2026-09-26 22:45:31,946", "event": "system.init"},
    {
        "asctime": "2026-09-26 22:45:37,218",
        "event": "post.generated",
        "bot_name": "Dan",
        "post_content": "Hello @Steve",
        "bot_model": "gemini-3.8-flash",
        "cost_usd": 0.0004,
        "latency_ms": 1200,
    },
    {
        "asctime": "2026-09-26 22:45:42,969",
        "event": "post.generated",
        "bot_name": "Steve",
        "post_content": "Hi @Dan, how are you? <script>x</script>",
        "bot_model": "gemini-3.8-flash",
        "cost_usd": 0.0006,
        "latency_ms": 900,
    },
    {"asctime": "2026-09-26 22:45:43,000", "event": "memory.form.success"},
    {"asctime": "2026-09-26 22:45:44,000", "event": "post.generation.fail"},
    {"asctime": "2026-09-26 22:45:51,219", "event": "sim.end"},
]


def write_log(path, rows=LOGS):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_report_written_next_to_log_with_cost(tmp_path):
    log = write_log(tmp_path / "run" / "simulation.jsonl")
    out = analyze_log(str(log))
    assert out == str(tmp_path / "run" / "analysis_report.html")
    html = open(out).read()
    assert "Simulation Analysis Report" in html and "Dan" in html and "Steve" in html
    assert "$0.0010" in html  # total cost
    assert "Failed turns" in html


def test_resolve_latest_and_folder(tmp_path):
    write_log(settings.RUNS_DIR / "sim_20260101_000000" / "simulation.jsonl")
    newest = write_log(settings.RUNS_DIR / "sim_20260926_000000" / "simulation.jsonl")
    assert resolve_log("latest") == newest
    assert resolve_log(str(newest.parent)) == newest


def test_cli_missing_log(capsys):
    assert analyze_cli("latest") == 1
    assert "Log not found" in capsys.readouterr().out


def test_no_posts(tmp_path, capsys):
    log = write_log(tmp_path / "empty.jsonl", [{"event": "system.init"}])
    assert analyze_log(str(log)) is None
    assert "No posts found" in capsys.readouterr().out
