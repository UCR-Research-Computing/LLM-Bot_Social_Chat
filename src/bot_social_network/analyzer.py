"""HTML report from a run's simulation.jsonl: activity, @mention graph, sentiment, cost."""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (headless servers, CI)
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402
from textblob import TextBlob  # type: ignore  # noqa: E402

from . import settings  # noqa: E402
from .ai_client import mentions as find_mentions  # noqa: E402


def resolve_log(arg: str) -> Path:
    """'latest', a run folder, or a .jsonl path -> the simulation.jsonl to read."""
    if arg == "latest":
        # Folder names are sim_YYYYMMDD_HHMMSS[_n], so name order is time order.
        runs = sorted(
            p
            for p in settings.RUNS_DIR.glob("sim_*")
            if (p / "simulation.jsonl").exists()
        )
        if not runs:
            raise FileNotFoundError(f"no runs in {settings.RUNS_DIR}")
        return runs[-1] / "simulation.jsonl"
    p = Path(arg).expanduser()
    if p.is_dir():
        p = p / "simulation.jsonl"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def analyze_cli(arg: str, output: str | None = None) -> int:
    try:
        log = resolve_log(arg)
    except FileNotFoundError as e:
        print(f"Log not found: {e}")
        return 1
    out = analyze_log(str(log), output)
    return 0 if out else 1


def analyze_log(log_file_path: str, output: str | None = None) -> str | None:
    print(f"Analyzing {log_file_path}...")

    data = []
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df_raw = pd.DataFrame(data)
    if df_raw.empty or "event" not in df_raw:
        print("No posts found in log file.")
        return None

    # Filter for posts
    posts_df = df_raw[df_raw["event"] == "post.generated"].copy()
    if posts_df.empty:
        print("No posts found in log file.")
        return None

    posts_df["asctime"] = pd.to_datetime(
        posts_df["asctime"], format="%Y-%m-%d %H:%M:%S,%f", errors="coerce"
    )

    # Basic Stats
    total_posts = len(posts_df)
    bot_names = posts_df["bot_name"].unique()
    bot_count = len(bot_names)

    duration = (posts_df["asctime"].max() - posts_df["asctime"].min()).total_seconds()
    duration_minutes = round(duration / 60, 2)

    posts_df["word_count"] = posts_df["post_content"].apply(
        lambda x: len(str(x).split())
    )
    avg_words_per_post = round(posts_df["word_count"].mean(), 2)

    # Bot Activity Table
    for col in ("cost_usd", "latency_ms", "tokens_in", "tokens_out"):
        if col not in posts_df:
            posts_df[col] = 0
        posts_df[col] = pd.to_numeric(posts_df[col], errors="coerce").fillna(0)
    if "bot_model" not in posts_df:
        posts_df["bot_model"] = ""
    bot_activity = (
        posts_df.groupby("bot_name")
        .agg(
            model=("bot_model", "last"),
            posts=("event", "count"),
            avg_words=("word_count", "mean"),
            avg_latency_s=("latency_ms", lambda s: s.mean() / 1000),
            cost_usd=("cost_usd", "sum"),
        )
        .round({"avg_words": 1, "avg_latency_s": 2, "cost_usd": 4})
        .reset_index()
    )
    total_cost = round(float(posts_df["cost_usd"].sum()), 4)
    errors = int((df_raw["event"] == "post.generation.fail").sum())
    memories = int((df_raw["event"] == "memory.form.success").sum())
    bot_activity_html = bot_activity.to_html(classes="table table-striped", index=False)

    # Mentions & Interaction Graph
    G: nx.DiGraph = nx.DiGraph()
    for bot in bot_names:
        G.add_node(bot)

    for _, row in posts_df.iterrows():
        sender = row["bot_name"]
        content = row["post_content"]
        for mention in find_mentions(str(content), list(bot_names)):
            if mention in bot_names and mention != sender:
                if G.has_edge(sender, mention):
                    G[sender][mention]["weight"] += 1
                else:
                    G.add_edge(sender, mention, weight=1)

    # Generate Interaction Graph Plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Normalize weights for edge widths
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    edge_widths: list[float] | float = 1.0
    if weights:
        max_weight = max(weights)
        edge_widths = [(w / max_weight) * 5 for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="skyblue", alpha=0.8)
    nx.draw_networkx_labels(
        G, pos, font_size=12, font_family="sans-serif", font_weight="bold"
    )
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
    )

    plt.title("Bot Interaction Graph (@mentions)", size=15)
    plt.axis("off")

    interaction_buf = BytesIO()
    plt.savefig(interaction_buf, format="png", bbox_inches="tight")
    interaction_graph_base64 = base64.b64encode(interaction_buf.getvalue()).decode(
        "utf-8"
    )
    plt.close()

    # Sentiment Analysis
    posts_df["sentiment"] = posts_df["post_content"].apply(
        lambda x: TextBlob(str(x)).sentiment.polarity
    )

    # Sentiment Plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Use rolling average for smoother plot
    window_size = max(1, total_posts // 10)
    posts_df["sentiment_smooth"] = (
        posts_df["sentiment"].rolling(window=window_size).mean()
    )

    sns.lineplot(
        data=posts_df, x="asctime", y="sentiment_smooth", marker="o", color="purple"
    )
    plt.title("Sentiment Trajectory Over Time", size=15)
    plt.xlabel("Time")
    plt.ylabel("Sentiment Polarity (Smooth)")
    plt.ylim(-1, 1)

    sentiment_buf = BytesIO()
    plt.savefig(sentiment_buf, format="png", bbox_inches="tight")
    sentiment_plot_base64 = base64.b64encode(sentiment_buf.getvalue()).decode("utf-8")
    plt.close()

    # Render Template
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir), autoescape=select_autoescape(["html"])
    )
    template = env.get_template("report_template.html")

    html_output = template.render(
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_posts=total_posts,
        duration_minutes=duration_minutes,
        avg_words_per_post=avg_words_per_post,
        bot_count=bot_count,
        bot_activity_table=bot_activity_html,
        interaction_graph_base64=interaction_graph_base64,
        sentiment_plot_base64=sentiment_plot_base64,
        total_cost=total_cost,
        errors=errors,
        memories=memories,
        source=str(log_file_path),
    )

    output_filename = output or os.path.join(
        os.path.dirname(os.path.abspath(log_file_path)), "analysis_report.html"
    )
    with open(output_filename, "w") as f:
        f.write(html_output)

    print(f"Analysis complete! Report saved to {output_filename}")
    return output_filename


if __name__ == "__main__":
    import sys

    sys.exit(analyze_cli(sys.argv[1] if len(sys.argv) > 1 else "latest"))
