import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from textblob import TextBlob  # type: ignore
import re
import base64
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
import argparse
from datetime import datetime
import os


def analyze_log(log_file_path):
    print(f"Analyzing {log_file_path}...")

    data = []
    with open(log_file_path, "r") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    df_raw = pd.DataFrame(data)

    # Filter for posts
    posts_df = df_raw[df_raw["event"] == "post.generated"].copy()
    if posts_df.empty:
        print("No posts found in log file.")
        return

    posts_df["asctime"] = pd.to_datetime(posts_df["asctime"])

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
    bot_activity = (
        posts_df.groupby("bot_name")
        .agg(posts=("event", "count"), avg_words=("word_count", "mean"))
        .round(2)
        .reset_index()
    )
    bot_activity_html = bot_activity.to_html(classes="table table-striped", index=False)

    # Mentions & Interaction Graph
    G = nx.DiGraph()
    for bot in bot_names:
        G.add_node(bot)

    for _, row in posts_df.iterrows():
        sender = row["bot_name"]
        content = row["post_content"]
        mentions = re.findall(r"@(\w+)", content)
        for mention in mentions:
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
    if weights:
        max_weight = max(weights)
        edge_widths = [(w / max_weight) * 5 for w in weights]
    else:
        edge_widths = 1

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
    env = Environment(loader=FileSystemLoader(template_dir))
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
    )

    output_filename = "analysis_report.html"
    with open(output_filename, "w") as f:
        f.write(html_output)

    print(f"Analysis complete! Report saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze simulation logs and generate an HTML report."
    )
    parser.add_argument("log_file", help="Path to the simulation.jsonl file.")
    args = parser.parse_args()

    if os.path.exists(args.log_file):
        analyze_log(args.log_file)
    else:
        print(f"Error: Log file not found at {args.log_file}")
