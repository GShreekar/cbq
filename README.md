# cbq (Codebase Q&A)

`cbq` is a local-first, privacy-respecting codebase semantic Q&A and analysis CLI tool. It uses `tree-sitter` for precise syntax parsing, `Ollama` for generating vector embeddings and local LLM chat responses, and an embedded `SQLite` database to create a lightning-fast search and chat experience right in your terminal. 

No code is ever uploaded to the cloud—everything runs entirely on your local machine.

---

## Features

- **Privacy First**: 100% local processing. No API keys, no cloud telemetry, and no code uploads.
- **Hybrid Code Search**: Finds code by meaning (embeddings) and by exact wording (SQLite FTS5/BM25), merging both rankings so a plain-English question and a bare identifier both work.
- **Code Chat (REPL)**: Talk directly to your codebase in an interactive chat session, maintaining context.
- **Git Diff Impact Analysis**: Pipe `git diff` to `cbq` to receive a detailed analysis of your changes, including potential bugs, dependencies, and affected components.
- **Intelligent Chunker**: Parses files with `tree-sitter` into logical units rather than line windows. Each chunk keeps the doc comments, decorators and attributes written above it; a type is indexed as a skeleton of its member signatures while each method is indexed on its own, so nothing is stored twice; functions assigned to a name (`const debounce = () => {}`) are indexed under that name; and every file gets a module chunk holding its imports, top-level constants and the list of symbols it defines.
- **Context-Enriched Embeddings**: What gets embedded is the code behind a header naming its file, language, symbol and enclosing type, so a method called `add` is not just the word `add` in a vacuum.
- **Model Auto-Provisioning**: Checks whether the configured models are on your Ollama server and pulls any that are missing, over Ollama's HTTP API, so Ollama running in Docker or on another machine works too.
- **History & Export**: Every question, its answer and the code it cited are recorded in that project's own index; review them with `cbq history` or export a Markdown transcript.

---

## Prerequisites

Before using `cbq`, ensure you have:

1. **Rust & Cargo** (v1.85 or newer, for the 2024 edition)
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

  cbq talks to Ollama over plain HTTP, so it builds without OpenSSL. To point `ollama.host` at an
  `https://` address, install with TLS support:

  ```bash
  cargo install cbq --features https
  ```

  Language grammars are downloaded on first use and cached under `~/.cache/tree-sitter-language-pack`.
  To compile them in instead, so cbq never fetches anything at runtime, name the ones you want at build time:

  ```bash
  TSLP_LANGUAGES=rust,python,javascript,typescript,go cargo install cbq
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

  Changing how cbq builds chunks, or switching embedding model, rebuilds the index automatically on the next run.

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
  Answers stream into the terminal as the model writes them. Piping the output (`cbq search "..." > notes.md`) writes plain text instead of terminal formatting.

  Search combines two methods: embedding similarity for meaning, and keyword matching for exact names, error codes and string literals. Both rankings are merged with Reciprocal Rank Fusion, so a chunk found by either surfaces, and one found by both rises to the top.

  Results are always the best matches available, with their similarity scores shown. Matches scoring below `search.similarity_threshold` are marked `(low confidence)` rather than hidden, so a weak-but-useful hit is never silently dropped.

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

### 4. History and Transcripts

History is per project and lives in that project's index, so it stays with the codebase it belongs to.

- **Show recent questions**:
  ```bash
  cbq history            # the 20 most recent, newest first
  cbq history --limit 5
  ```

- **Export a transcript**:
  Writes every question, the answer given, and the code each answer cited, grouped into the sessions they were asked in.
  ```bash
  cbq export                       # into ~/.cbq/exports/
  cbq export --output notes.md
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
  cbq config set ollama.parallelism 4
  ```

#### Default Configurations
```toml
[ollama]
host = "http://localhost"
port = 11434
embedding_model = "nomic-embed-text"
chat_model = "qwen2.5:1.5b"

parallelism = 1

[search]
top_k = 5
similarity_threshold = 0.5
```

`ollama.parallelism` is how many embedding requests cbq keeps in flight while indexing. One suits a
server that works through requests serially, which is the default; raise it only if you have set
`OLLAMA_NUM_PARALLEL` above 1 and have the GPU memory for it.

`similarity_threshold` marks weak matches in search results rather than hiding them; in `cbq analyze` it does filter, so unrelated code is kept out of the review.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
