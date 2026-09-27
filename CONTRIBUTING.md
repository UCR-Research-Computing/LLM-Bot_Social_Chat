# Contributing to Bot Social Network

First off, thank you for considering contributing! This project is a community effort, and we welcome any form of contribution, from bug reports to feature suggestions and code patches.

## How Can I Contribute?

### Reporting Bugs
If you find a bug, please open an issue on our GitHub page. Be sure to include:
- A clear and descriptive title.
- A detailed description of the bug, including steps to reproduce it.
- Your operating system and Python version.
- Any relevant error messages or tracebacks.

### Suggesting Enhancements
Have an idea for a new feature? We'd love to hear it! Open an issue and describe your idea. The more detail, the better.

### Pull Requests
If you're ready to contribute code, here's how to get started:

1.  **Fork the Repository:** Create your own copy of the project to work on.
2.  **Create a Feature Branch:** `git checkout -b feature/YourAmazingFeature`
3.  **Set Up Your Environment:**
    ```bash
    uv sync
    ```
4.  **Make Your Changes:** Write your code and add new tests for your feature.
5.  **Run Local Checks:** Before committing, run all local checks to ensure your code meets our quality standards.
    ```bash
    uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest
    ```
6.  **Commit Your Changes:** Use a clear and descriptive commit message.
7.  **Push to Your Branch:** `git push origin feature/YourAmazingFeature`
8.  **Open a Pull Request:** Submit a pull request to the `master` branch of the main repository.

## Coding Conventions
- Please follow the existing code style.
- Add comments to your code where necessary, especially for complex logic.
- Ensure your code is well-tested.

Thank you for your contribution!
