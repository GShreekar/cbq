# Changelog

All notable changes to cbq are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-09-22

The first release since 1.0.1, adding twelve commands and rebuilding retrieval, indexing and
output. Everything runs against Ollama on your own machine, as before.

### Added

**Serving other tools**
- `cbq mcp --stdio` serves the index over the Model Context Protocol, speaking JSON-RPC 2.0 on
  stdin and stdout, so Claude Code, Cursor or Zed can query a codebase without reading it into
  their context. It offers `search_code`, `find_definition`, `find_references` and
  `project_status`. Models are never pulled in this mode, and nothing but JSON-RPC reaches stdout.
- `--json` on every command, with an exit-code contract: failures print `{"error": ...}` on stderr
  and exit non-zero, so scripts and editors can tell success from failure without parsing prose.

**Following symbols**
- A call and import graph is recorded while indexing, backing `cbq def`, `cbq refs` and
  `cbq callers`. These read SQLite only: no model, no embeddings, and an immediate answer.
- `cbq explain <file>` explains what a file is for; `cbq explain <file>:<line>` finds the smallest
  indexed symbol covering that line and pulls in its callers and callees.

**Searching**
- Hybrid search: embedding similarity for meaning and SQLite FTS5/BM25 for exact wording, merged
  with Reciprocal Rank Fusion, so a plain-English question and a bare identifier both work.
- Listwise reranking by the chat model, over a wider candidate set than is returned.
- `cbq search --all` searches every indexed project at once and merges the results into one
  ranking. Indexes built with a different embedding model are excluded and named with the reason,
  because vectors from different models cannot be compared.

**Trusting the answer**
- Every `file:line` an answer cites is checked against the index. Citations naming a file that was
  never indexed, or a line past the end of one, are reported in the terminal and appear as
  `unverified_citations` in `--json`.

**Keeping the index current**
- Incremental indexing: only files whose contents changed are re-parsed and re-embedded, tracked
  by content hash, with `--force` to rebuild from scratch.
- `cbq watch` indexes once, then re-indexes on change, debounced and ignoring build directories.
- `cbq status` summarises the index and how far it has drifted from the files on disk;
  `cbq doctor` checks Ollama reachability, model and dimension agreement, index age and
  stored-data permissions, exiting non-zero when something is broken.
- `cbq list` and `cbq clean` show and remove the indexes cbq has built.

**Configuration and shell integration**
- A per-project `.cbq.toml`, found in the project or any directory above it, overrides the global
  configuration for work inside that project. The Ollama address is deliberately *not*
  overridable: a `.cbq.toml` arrives with code you cloned, and a repository must not be able to
  redirect where your source code is sent.
- `cbq completions <shell>` for bash, zsh, fish, elvish and PowerShell, and `cbq man`, both
  generated from the command definitions so they cannot drift from `--help`.

### Changed

- Chunking is AST-shaped: a chunk keeps the doc comments, decorators and attributes written above
  it; a type is indexed as a skeleton of member signatures while each method is indexed
  separately; functions assigned to a name are indexed under that name; and every file gets a
  module chunk holding its imports, top-level constants and the symbols it defines.
- What gets embedded is the code behind a header naming its file, language, symbol and enclosing
  type, so a method called `add` is not just the word `add` in a vacuum.
- Indexing reads and parses files in parallel and embeds them in batches, with
  `ollama.parallelism` controlling concurrency.
- Embedding now uses Ollama's batch `/api/embed` endpoint.
- Answers stream into the terminal as the model writes them, with Markdown rendering, progress
  spinners and file-type badges. Piping the output writes plain text instead.
- The chat REPL supports line editing and history: arrow keys, Ctrl-A/E, Ctrl-W, Ctrl-C to abandon
  a line and Ctrl-D to end the session.
- `cbq analyze` reviews a git diff for bugs and names the call sites a change may break, using the
  symbol graph.
- Questions, answers and cited code are recorded per project, and `cbq export` writes a Markdown
  transcript. A `--all` search is not recorded, since its answer belongs to no single project.
- Release builds use LTO, one codegen unit and stripped symbols.
- Index and chunk formats are versioned: cbq notices an index written by an older format and
  rebuilds it rather than returning wrong results.

### Security

- Files that look like credentials, and source files containing private keys or access tokens, are
  kept out of the index and listed so you know what was skipped. `--allow-secrets` overrides this.
- `~/.cbq` and the indexes under it are created readable only by their owner, since an index holds
  your source code. `cbq doctor` warns if the permissions have been widened.
- Pointing `ollama.host` at a non-local address is refused unless explicitly allowed, and warns on
  every run once it is.
- Retrieved code is fenced in the prompt and marked as data to be explained rather than
  instructions to follow.
- `.cbqignore`, `--include`/`--exclude` and a per-file size cap control what is indexed.

### Fixed

- Searching from a subdirectory now uses the nearest indexed directory at or above it, and `-C`
  selects a different project, instead of assuming the current directory.
- `cbq watch` no longer re-indexes in a loop: file access events are ignored, and only content and
  name changes count.
- Counting rows in the full-text index no longer reports the source table's count.

### Notes

- Minimum supported Rust version is now declared as 1.88, taken from the locked dependencies.
- cbq 1.1.0 reads indexes built by 1.0.1 by rebuilding them; run `cbq index` once after upgrading.

## [1.0.1] - 2026-06-25

- Model auto-provisioning over Ollama's HTTP API, CLI usability improvements, project
  documentation and license.

<!-- 1.0.1 was published to crates.io but never tagged in git; c6ee241 is that release. -->
[1.1.0]: https://github.com/GShreekar/cbq/compare/c6ee241...v1.1.0
[1.0.1]: https://github.com/GShreekar/cbq/commit/c6ee241
