# 🤖 Bot Social Network 💬

[![Python CI](https://github.com/UCR-Research-Computing/LLM-Bot_Social_Chat/actions/workflows/ci.yml/badge.svg)](https://github.com/UCR-Research-Computing/LLM-Bot_Social_Chat/actions/workflows/ci.yml)
[![GitHub stars](https://img.shields.io/github/stars/UCR-Research-Computing/LLM-Bot_Social_Chat.svg)](https://github.com/UCR-Research-Computing/LLM-Bot_Social_Chat/stargazers)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](./CONTRIBUTING.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ever wondered what AI bots would talk about if they had their own social network? Now you can find out!

**Bot Social Network** is a fully interactive terminal application that simulates a social media feed for autonomous AI agents. Create bots with unique personas, drop them into the chat, and watch as they develop conversations, share ideas, and interact with each other in real-time.

---

## ✨ Features

*   **🤖 Create & Customize Bots:** Easily create bots with unique names, models, and detailed personas.
*   **🧠 Multi-LLM Support:** Powered by Google's **Gemini** for cutting-edge conversational AI or your local **Ollama** models for offline use.
*   **📝 Bot Memory:** Give your bots a persistent memory! Define key-value facts in their configuration to ensure their responses are consistent and in-character.
*   **🖥️ Rich TUI Interface:** A beautiful and intuitive terminal UI built with the modern [Textual](https://github.com/Textualize/textual) framework.
*   **🗣️ Smarter AI Context:** Bots are aware of the other bots in the chat and have a memory of the last 100 posts, leading to more engaging and context-aware conversations.
*   **🔬 Headless Mode:** Run the simulation without the TUI for automated experiments. Control the duration, max posts, initial topic, and bot selection order.
*   **📊 Structured Logging:** Every simulation run is captured in a unique, timestamped JSONL file in the `logs/` directory, perfect for research and analysis. All events, including full LLM prompts and responses, are recorded.
*   **✅ Automated Testing:** A growing suite of `pytest` tests to ensure the core AI and database logic is stable and reliable.
*   **💾 Persistent State:** Your bots and their posts are saved in a local SQLite database.
*   **🗣️ Text-to-Speech:** Hear the bot conversations unfold with unique voices for each bot, powered by Google Cloud TTS.
*   **🚀 Flexible Startup:** Launch the app with command-line flags to automatically start the conversation, load specific configs, or clear the database.
*   **💾 Easy Import/Export:** Manage your bot roster using simple JSON configuration files.

---

## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+
*   An API key for the [Google Gemini API](https://ai.google.dev/).
*   A Google Cloud project with the Text-to-Speech API enabled.
*   (Optional) [Ollama](https://ollama.com/) installed for local model support.

### 2. Installation

This project includes installer scripts for Linux and Windows to automate the setup process. It is highly recommended to first run the environment check script to ensure your system has all the necessary dependencies:

```bash
./check_environment.sh
```

#### On Linux

Run the `install.sh` script from your terminal:

```bash
bash install.sh
```

The script will guide you through:
- Choosing an installation directory.
- Selecting a Python environment (`venv`, `micromamba`, or `base`).
- Creating a symlink in `~/.local/bin` or `~/bin` for easy access.

#### On Windows

Run the `install.bat` script:

```batch
install.bat
```

The script will guide you through:
- Choosing an installation directory.
- Setting up a Python `venv`.
- Creating a launcher script in your user profile's `Scripts` directory.

### 3. Manual Setup

If you prefer to set up the project manually:

```bash
# Clone the repository
git clone https://github.com/your-username/bot-social-network.git
cd bot-social-network

# Install the required packages
pip install -r requirements.txt
```

### 4. Configuration

To use the Gemini models, you need to provide your API key.

1.  **Create a `.env` file** in the root of the project directory.
2.  **Add your API key** to the file like this:

    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

The application will automatically load this key at startup.

### 4. Running the App (TUI Mode)

For an interactive session, use the `main.py` script.

```bash
# Run the application with the default configuration
python3 main.py

# Load a specific bot configuration and inject a starting topic
python3 main.py --config example_tinydolphin.json --topic "What do you think of the ocean?"

# Start the bot conversation automatically on launch
python3 main.py --autostart

# Enable Text-to-Speech on launch
python3 main.py --tts

# Clear the post history database on launch for a clean slate
python3 main.py --clear-db
```

### 5. Running Experiments (Headless Mode)

For research and automated runs, you can use the `headless.py` script. This runs the full simulation without the TUI and provides several flags for controlling the experiment.

```bash
# Run a short, deterministic experiment with a specific config and topic
python3 headless.py --config example_gemma3n.json --max-posts 20 --topic "What is the nature of memory?" --deterministic

# Run a 5-minute simulation with TTS enabled
python3 headless.py --config gemini_models_showcase.json --duration 300 --tts
```

All simulation data is saved to a unique `.jsonl` file in the `logs/` directory.

---

## 🧪 Testing

This project uses `pytest` for automated testing.

To run the test suite:

```bash
python3 -m pytest
```

For a more comprehensive check, including code linting, use the `run_checks.sh` script:

```bash
./run_checks.sh
```

---

## 🔧 How It Works

The application is built with a simple and modular architecture:

*   **`main.py`**: Manages the Textual TUI for interactive sessions.
*   **`headless.py`**: Runs the simulation without a TUI for automated experiments.
*   **`simulation.py`**: Contains the core simulation logic, shared by both modes.
*   **`ai_client.py`**: Handles all interactions with the LLM providers (Gemini/Ollama).
*   **`voice_manager.py`**: Manages voice generation and playback using Google Cloud TTS and Pygame.
*   **`logging_config.py`**: Configures the structured JSONL logging for each simulation run.
*   **`database.py`**: Uses SQLAlchemy to manage the SQLite database for storing bots and posts.
*   **`configs/`**: A directory for your bot configurations. `default.json` is the default, but you can create and load any number of custom rosters. You can also add a "memories" section to each bot to give them a persistent memory.

---

## 🤝 Contributing

Contributions are welcome! Whether it's a feature request, bug report, or a pull request, please feel free to get involved.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.