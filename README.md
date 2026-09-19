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
- **Model Auto-Provisioning**: Checks whether the configured models are on your Ollama server and pulls any that are missing, over Ollama's HTTP API, so Ollama running in Docker or on another machine works too.
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
  *Note: The SQLite database is stored locally inside your home directory under `~/.cbq/codebases/<project-name>-<path-hash>/db.sqlite`, so projects that share a folder name keep separate indexes.*

  Re-running `cbq index` only embeds files whose content changed since the last run, and drops files that were deleted; an unchanged project is up to date in well under a second, without contacting Ollama. Each file is saved as soon as it is embedded, so an interrupted run picks up where it stopped. Chunks that fail to embed are skipped, listed, and retried on the next run. Use `--force` to rebuild everything; changing `ollama.embedding_model` triggers a rebuild automatically.

  Indexing honors `.gitignore` (even outside a git repository), `.ignore`, and `.cbqignore` files, which use the same syntax for paths you want kept out of the index. It always skips version-control, dependency and build directories (`.git`, `node_modules`, `vendor`, `target`, `dist`, `build`, virtualenvs and similar), lockfiles, and minified `*.min.*` files. Files over 512 KB are skipped and listed; raise the limit with `cbq index --max-file-size <KB>`.

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
  The number of returned chunks defaults to `search.top_k` from your config; override it with `--limit`:
  ```bash
  cbq search "database initialization" --limit 3
  ```
  Queries work from any subdirectory: `cbq` uses the nearest indexed directory at or above where you run it. To query a different project, pass `-C`:
  ```bash
  cbq search "database initialization" -C ~/code/other-project
  ```

- **Interactive Chat REPL**:
  Start a persistent chat session to explore your codebase interactively.
  ```bash
  cbq chat
  ```
  The session remembers the conversation, so follow-ups like "what calls it?" refer back to earlier answers. Type `/clear` to start a new conversation, and `exit`, `quit` or `Ctrl-D` to leave.

---

### 3. Git Integration & Impact Analysis

- **Review Changes**:
  `cbq analyze` has the local chat model review a diff for likely bugs and for code elsewhere in the project that depends on what changed. Related code is found by searching the index with each changed hunk; without an index, the diff is still reviewed on its own.
  ```bash
  cbq analyze                 # all uncommitted changes (git diff HEAD)
  cbq analyze --staged        # only what's staged
  cbq analyze --base main     # everything since this branch left main, including uncommitted work
  git diff v1.2 | cbq analyze # any diff piped in
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
