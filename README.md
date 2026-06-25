# cbq (Codebase Q&A)

`cbq` is a local-first, privacy-respecting codebase semantic Q&A and analysis CLI tool. It uses `tree-sitter` for precise syntax parsing, `Ollama` for generating vector embeddings and local LLM chat responses, and an embedded `SQLite` database to create a lightning-fast search and chat experience right in your terminal. 

No code is ever uploaded to the cloud—everything runs entirely on your local machine.

---

## Features

- **Privacy First**: 100% local processing. No API keys, no cloud telemetry, and no code uploads.
- **Semantic Code Search**: Find contextually and semantically relevant code using natural language.
- **Code Chat (REPL)**: Talk directly to your codebase in an interactive chat session, maintaining context.
- **Git Diff Impact Analysis**: Pipe `git diff` to `cbq` to receive a detailed analysis of your changes, including potential bugs, dependencies, and affected components.
- **Intelligent Chunker**: Parses files using `tree-sitter` AST to chunk code cleanly into logical units (functions, classes, structures, modules) rather than naive line splitting.
- **Model Auto-Provisioning**: Checks if configured models are downloaded on your local Ollama instance and pulls them automatically if missing.
- **History & Export**: Review your query history or export chat transcripts to Markdown for documentation.

---

## Prerequisites

Before using `cbq`, ensure you have:

1. **Rust & Cargo** (v1.70 or newer)
2. **Ollama** installed and running on your system.
   - [Download Ollama here](https://ollama.com/)
   - Make sure the Ollama server is running (usually on `http://localhost:11434`).
   - The default models configured are **`nomic-embed-text`** (for embeddings) and **`qwen2.5:1.5b`** (for chat). If they are not present, `cbq` will attempt to auto-pull them on the first run.

---

## Installation

### Via Cargo (Recommended)

You can install `cbq` directly from [crates.io](https://crates.io):

```bash
cargo install cbq
```

### From Source

Alternatively, you can compile and install `cbq` locally:

```bash
# Clone the repository
git clone https://github.com/GShreekar/cbq.git
cd cbq

# Build and install the binary
cargo install --path .
```

Ensure your cargo binary directory (usually `~/.cargo/bin`) is added to your system's `PATH`.

---

## CLI Usage Guide

`cbq` provides a range of commands for codebase ingestion, search, chat, configuration, and analysis.

### 1. Ingesting & Indexing

To get started, you must index your codebase.

- **Scanning & Estimating**:
  Scan your directory to see what file extensions are present and get an estimate of indexable files.
  ```bash
  cbq init [path/to/project]
  ```
  *(Defaults to current directory `./` if no path is provided)*

- **AST Parsing Check**:
  Scan the directory, run the AST parsers, and print out all discovered logical code chunks.
  ```bash
  cbq parse [path/to/project]
  ```

- **Database Ingestion (Indexing)**:
  Run the file parser, request embeddings from Ollama, and save chunks to the SQLite database.
  ```bash
  cbq index [path/to/project]
  ```
  *Note: The SQLite database is stored locally inside your home directory under `~/.cbq/codebases/<project-name>/embeddings.db`.*

---

### 2. Querying & Chatting

Once the codebase is indexed, you can run queries.

- **One-off Semantic Search & Q&A**:
  Query your codebase directly. `cbq` will fetch relevant code chunks, print them out with syntax highlighting, and feed them to the local LLM to generate an explanation.
  ```bash
  cbq search "How is the AST traversed?"
  ```
  You can also pass queries as the default argument if no subcommand is supplied:
  ```bash
  cbq "How is the AST traversed?"
  ```
  Change the number of returned chunks using `--limit`:
  ```bash
  cbq search "database initialization" --limit 3
  ```

- **Interactive Chat REPL**:
  Start a persistent chat session to explore your codebase interactively.
  ```bash
  cbq chat
  ```
  Type `exit` or `quit` to exit the chat mode.

---

### 3. Git Integration & Impact Analysis

- **Analyze Diffs**:
  Analyze active git diffs to detect what code was added or changed, find semantically related context files in your repository, and print an impact report.
  ```bash
  git diff | cbq analyze
  ```

---

### 4. History and Logs

- **Display History**:
  Shows a timestamped list of all queries you have run along with the number of matched code snippets.
  ```bash
  cbq history
  ```

- **Export Logs**:
  Export all history logs to a Markdown file in your configuration directory.
  ```bash
  cbq export
  ```

---

### 5. Configuration Settings

Manage your configurations globally. Configurations are stored inside `~/.cbq/config.toml`.

- **Initialize Config**:
  Create the default configuration file.
  ```bash
  cbq config init
  ```
- **Display Configurations**:
  Print out all current config parameters in TOML format.
  ```bash
  cbq config get
  ```
- **Update Parameters**:
  Modify config values directly.
  ```bash
  cbq config set ollama.chat_model "llama3"
  cbq config set search.top_k 10
  cbq config set search.similarity_threshold 0.6
  ```

#### Default Configurations
```toml
[ollama]
host = "http://localhost"
port = 11434
embedding_model = "nomic-embed-text"
chat_model = "qwen2.5:1.5b"

[search]
top_k = 5
similarity_threshold = 0.5
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
