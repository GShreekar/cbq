pub mod cli;
pub mod services;
pub mod db;
pub mod config;
pub mod ui;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::Context;
use clap::Parser;
use cli::args::{Cli, Commands};
use services::chunker::CodeChunk;
use services::file_discovery::{discover_files, SkipReason, SkippedFile, DEFAULT_MAX_FILE_BYTES};
use services::index_plan::{plan_index, read_source_file, IndexPlan, SourceFile};
use services::parser::{parse_file, parse_source};
use services::vector_search::{search_codebase, SearchResult};
use services::chat_history::{save_chat, get_history, export_history_to_markdown};
use services::ollama::{is_same_model, Ollama};
use services::git::{find_repository_root, parse_diff, read_git_diff, DiffSource, FileChange, FileDiff, Hunk};
use services::review::{
    build_review_prompt, describe_change, is_changed_code, merge_related_code, to_diff_path, to_index_path,
};
use config::settings::{load_config, Config};
use db::schema::{init_db, open_index};
use db::location::{canonical_project_root, find_indexed_project, find_legacy_index, index_path_for, IndexedProject};
use db::index_metadata::{ensure_index_model_matches, read_embedding_model, write_embedding_model};
use db::queries::{
    delete_untracked_chunks, get_db_stats, has_chunks, read_file_hashes, remove_files, replace_file_chunks,
    reset_index, FileUpdate,
};
use indicatif::{ProgressBar, ProgressStyle};
use colored::Colorize;

// A run of failures this long means Ollama or the model is broken, not individual chunks.
const MAX_CONSECUTIVE_EMBEDDING_FAILURES: usize = 10;
const MAX_SKIPPED_ITEMS_LISTED: usize = 5;
const HISTORY_TURNS_IN_PROMPT: usize = 3;
// Long answers are cut so a few turns of history can't crowd the code context out of the prompt.
const MAX_ANSWER_CHARS_IN_HISTORY: usize = 1_500;
// Each hunk costs one embedding call, about a second on CPU, so very large diffs are sampled.
const MAX_HUNKS_SEARCHED: usize = 20;
// More than will be kept, because hits on the changed code itself are filtered out afterwards.
const RELATED_RESULTS_PER_HUNK: usize = 6;
const MAX_RELATED_CHUNKS_IN_REVIEW: usize = 4;

struct ChatTurn {
    question: String,
    answer: String,
}

// What answering a question needs: settings, the model server, and the project's index.
struct Session<'a> {
    config: &'a Config,
    ollama: &'a Ollama,
    conn: &'a rusqlite::Connection,
}

#[derive(Default)]
struct EmbeddedChunks {
    chunks: Vec<CodeChunk>,
    embeddings: Vec<Vec<f32>>,
    skipped: Vec<String>,
}

struct ParsedFile {
    relative_path: String,
    content_hash: String,
    chunks: Vec<CodeChunk>,
    parse_failed: bool,
}

#[derive(Default)]
struct IndexOutcome {
    stored_chunks: usize,
    stored_files: usize,
    skipped_chunks: Vec<String>,
}

struct ChunkEmbedder<'a> {
    ollama: &'a Ollama,
    embedding_model: &'a str,
    progress: ProgressBar,
    consecutive_failures: usize,
}

#[tokio::main]
async fn main() {
    let args = Cli::parse();

    match args.command {
        Some(Commands::Init { path }) => {
            println!("{}", "Scanning repository...".cyan());

            let spinner = ProgressBar::new_spinner();
            spinner.enable_steady_tick(Duration::from_millis(100));
            spinner.set_message("Discovering files...");

            match discover_files(&path, DEFAULT_MAX_FILE_BYTES) {
                Ok(result) => {
                    spinner.finish_and_clear();

                    println!(
                        "{} {} {}",
                        "Found".green().bold(),
                        result.files.len().to_string().yellow().bold(),
                        "indexable files:".green().bold()
                    );

                    let mut counts: Vec<(&String, &usize)> = result.extension_counts.iter().collect();
                    counts.sort_by(|a, b| b.1.cmp(a.1));

                    for (ext, count) in counts {
                        println!(
                            "  - {} (.{ext}): {} files",
                            ext.to_uppercase().blue(),
                            count.to_string().bold()
                        );
                    }

                    println!();
                    print_skipped_files(&result.skipped, &path);
                    println!("{}", "✓ Ready to index. Run: cbq index <path>".green());
                }
                Err(err) => {
                    spinner.finish_and_clear();
                    eprintln!("{} {}", "Error scanning files:".red().bold(), err);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Parse { path }) => {
            println!("{}", "Scanning and parsing repository...".cyan());
            
            let spinner = ProgressBar::new_spinner();
            spinner.enable_steady_tick(Duration::from_millis(100));
            spinner.set_message("Parsing files...");

            match discover_files(&path, DEFAULT_MAX_FILE_BYTES) {
                Ok(discovery) => {
                    let mut total_chunks = 0;

                    for file in discovery.files {
                        match parse_file(&file) {
                            Ok(chunks) => {
                                total_chunks += chunks.len();
                                if !chunks.is_empty() {
                                    println!(
                                        "\n{} {}",
                                        "File:".magenta().bold(),
                                        file.display().to_string().underline()
                                    );
                                    for chunk in chunks {
                                        println!(
                                            "  [{}] name: '{}' (lines {}-{})",
                                            chunk.chunk_type.yellow(),
                                            chunk.name.blue().bold(),
                                            chunk.start_line,
                                            chunk.end_line
                                        );
                                    }
                                }
                            }
                            Err(err) => {
                                eprintln!(
                                    "{} Failed to parse {}: {}",
                                    "Error:".red().bold(),
                                    file.display(),
                                    err
                                );
                            }
                        }
                    }
                    spinner.finish_and_clear();
                    println!(
                        "\n{} Total logical chunks found: {}",
                        "✓ Parsing completed.".green().bold(),
                        total_chunks.to_string().bold().yellow()
                    );
                }
                Err(err) => {
                    spinner.finish_and_clear();
                    eprintln!("{} {}", "Error:".red().bold(), err);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Index { path, force, max_file_size_kb }) => {
            if let Err(err) = run_index(&path, force, max_file_size_kb * 1024).await {
                eprintln!("{} {:#}", "Error:".red().bold(), err);
                std::process::exit(1);
            }
        }
        Some(Commands::Search { query, limit, directory }) => {
            run_search(&query, limit, &directory).await;
        }
        Some(Commands::Config { action }) => {
            match action {
                cli::args::ConfigAction::Init => {
                    let path = config::settings::get_config_path().unwrap();
                    let default_conf = config::settings::Config::default();
                    if let Err(e) = config::settings::save_config(&default_conf) {
                        eprintln!("Failed to save config: {}", e);
                    } else {
                        println!("Created {} with defaults", path.to_string_lossy());
                    }
                }
                cli::args::ConfigAction::Get => {
                    match config::settings::load_config() {
                        Ok(conf) => {
                            let toml_str = toml::to_string(&conf).unwrap();
                            println!("{}", toml_str);
                        }
                        Err(e) => eprintln!("Failed to load config: {}", e),
                    }
                }
                cli::args::ConfigAction::Set { key, value } => {
                    let mut conf = match config::settings::load_config() {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Failed to load config: {}", e);
                            std::process::exit(1);
                        }
                    };

                    match key.as_str() {
                        "ollama.host" => conf.ollama.host = value,
                        "ollama.port" => {
                            if let Ok(v) = value.parse::<u16>() {
                                conf.ollama.port = v;
                            } else {
                                eprintln!("Error: port must be an integer");
                                std::process::exit(1);
                            }
                        }
                        "ollama.embedding_model" => {
                            if value != conf.ollama.embedding_model {
                                println!(
                                    "{}",
                                    "Note: existing indexes use the previous model; rebuild them with `cbq index`.".yellow()
                                );
                            }
                            conf.ollama.embedding_model = value;
                        }
                        "search.top_k" | "top_k" => {
                            if let Ok(v) = value.parse::<usize>() {
                                conf.search.top_k = v;
                            } else {
                                eprintln!("Error: top_k must be an integer");
                                std::process::exit(1);
                            }
                        }
                        "search.similarity_threshold" | "similarity_threshold" => {
                            if let Ok(v) = value.parse::<f64>() {
                                conf.search.similarity_threshold = v;
                            } else {
                                eprintln!("Error: similarity_threshold must be a float");
                                std::process::exit(1);
                            }
                        }
                        _ => {
                            eprintln!("Unknown config key: '{}'", key);
                            std::process::exit(1);
                        }
                    }

                    if let Err(e) = config::settings::save_config(&conf) {
                        eprintln!("Failed to save config: {}", e);
                    } else {
                        println!("{}", "✓ Updated config".green());
                    }
                }
            }
        }
        Some(Commands::History) => {
            match get_history() {
                Ok(history) => {
                    if history.is_empty() {
                        println!("No search history found.");
                        return;
                    }
                    crate::ui::formatter::print_section("Search Query History Log");
                    for entry in history {
                        println!(
                            "{} - \"{}\" ({} matches)",
                            entry.timestamp.dimmed(),
                            entry.query.bold().yellow(),
                            entry.results.len()
                        );
                    }
                }
                Err(err) => eprintln!("Failed to retrieve history: {}", err),
            }
        }
        Some(Commands::Export) => {
            match export_history_to_markdown() {
                Ok(path) => {
                    crate::ui::formatter::print_success_msg(&format!(
                        "Exported search history to: {}",
                        path.to_string_lossy().underline().yellow()
                    ));
                }
                Err(err) => eprintln!("Failed to export history: {}", err),
            }
        }
        Some(Commands::Chat { directory }) => {
            if let Err(err) = run_chat_repl(&directory).await {
                eprintln!("{} Chat session error: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }
        }
        Some(Commands::Analyze { staged, base, directory }) => {
            let requested_source = match (staged, base) {
                (true, _) => Some(DiffSource::Staged),
                (false, Some(base)) => Some(DiffSource::SinceBase(base)),
                (false, None) => None,
            };
            if let Err(err) = run_analyze(&directory, requested_source).await {
                eprintln!("{} Analysis failed: {:#}", "Error:".red().bold(), err);
                std::process::exit(1);
            }
        }
        None => {
            if let Some(query) = args.default_query {
                run_search(&query, None, Path::new(".")).await;
            } else {
                println!("No arguments provided. Run with --help to see usage.");
            }
        }
    }
}

async fn run_index(path: &Path, force: bool, max_file_bytes: u64) -> Result<(), anyhow::Error> {
    let project_root = canonical_project_root(path)?;
    println!("Indexing {}...", project_root.display().to_string().cyan());

    let config = match load_config() {
        Ok(c) => c,
        Err(_) => {
            let default_conf = crate::config::settings::Config::default();
            let _ = crate::config::settings::save_config(&default_conf);
            default_conf
        }
    };

    let db_path = index_path_for(&project_root)?;
    let mut conn = init_db(&db_path).context("Failed to initialize database")?;
    let rebuild_reason = rebuild_reason(&conn, &config, force)?;

    let discovery = discover_files(&project_root, max_file_bytes).context("File discovery failed")?;
    let (source_files, unreadable_files) = read_source_files(&discovery.files, &project_root);
    let skipped_files: Vec<SkippedFile> = discovery.skipped.into_iter().chain(unreadable_files).collect();
    print_skipped_files(&skipped_files, &project_root);

    let recorded_hashes = match rebuild_reason {
        Some(_) => HashMap::new(),
        None => read_file_hashes(&conn)?,
    };
    let plan = plan_index(source_files, &recorded_hashes);
    print_index_plan(&plan, rebuild_reason.as_deref());

    if rebuild_reason.is_none() {
        remove_files(&mut conn, &plan.removed_paths)?;
    }
    if rebuild_reason.is_none() && plan.files_to_index().next().is_none() {
        delete_untracked_chunks(&conn)?;
        println!("{} Index is up to date\n", "✓".green().bold());
        print_index_statistics(&conn, &db_path);
        return Ok(());
    }

    println!("Checking Ollama connection...");
    let embedding_model = &config.ollama.embedding_model;
    let ollama = prepare_ollama(&config, &[embedding_model]).await?;
    println!("{} Connected to {}", "✓".green().bold(), ollama.address());
    println!("Embedding model: {}\n", embedding_model.yellow().bold());

    // The old index is only cleared once Ollama is known to be ready to rebuild it.
    match rebuild_reason {
        Some(_) => reset_index(&mut conn, embedding_model)?,
        None => write_embedding_model(&conn, embedding_model)?,
    }

    println!("Parsing files...");
    let parsed_files = parse_planned_files(&plan);
    println!();

    println!("Generating embeddings...");
    let outcome = embed_and_store(&ollama, embedding_model, &mut conn, parsed_files).await?;
    delete_untracked_chunks(&conn)?;
    println!();
    print_skipped_chunks(&outcome.skipped_chunks);

    println!(
        "{} {} chunks stored from {} files\n",
        "✓".green().bold(),
        outcome.stored_chunks.to_string().yellow().bold(),
        outcome.stored_files
    );
    print_index_statistics(&conn, &db_path);
    Ok(())
}

// A full rebuild is needed when asked for, or when the stored vectors came from a different or unknown model.
fn rebuild_reason(conn: &rusqlite::Connection, config: &Config, force: bool) -> Result<Option<String>, anyhow::Error> {
    if force {
        return Ok(Some("--force was given".to_string()));
    }
    let configured_model = &config.ollama.embedding_model;
    let reason = match read_embedding_model(conn)? {
        Some(indexed_model) if !is_same_model(&indexed_model, configured_model) => Some(format!(
            "the embedding model changed from {} to {}",
            indexed_model, configured_model
        )),
        Some(_) => None,
        None if has_chunks(conn)? => Some("the index was built by an older cbq version".to_string()),
        None => None,
    };
    Ok(reason)
}

fn read_source_files(paths: &[PathBuf], project_root: &Path) -> (Vec<SourceFile>, Vec<SkippedFile>) {
    let mut source_files = Vec::new();
    let mut unreadable_files = Vec::new();
    for path in paths {
        match read_source_file(path, project_root) {
            Ok(source_file) => source_files.push(source_file),
            Err(err) => unreadable_files.push(SkippedFile {
                path: path.clone(),
                reason: SkipReason::Unreadable(err.to_string()),
            }),
        }
    }
    (source_files, unreadable_files)
}

fn print_index_plan(plan: &IndexPlan, rebuild_reason: Option<&str>) {
    if let Some(reason) = rebuild_reason {
        println!(
            "Rebuilding the whole index ({} files) because {}.",
            plan.new_files.len().to_string().yellow().bold(),
            reason
        );
        return;
    }
    println!(
        "Changes since the last index: {} new, {} changed, {} unchanged, {} removed",
        plan.new_files.len().to_string().yellow().bold(),
        plan.changed_files.len().to_string().yellow().bold(),
        plan.unchanged_count,
        plan.removed_paths.len()
    );
}

fn parse_planned_files(plan: &IndexPlan) -> Vec<ParsedFile> {
    let files: Vec<&SourceFile> = plan.files_to_index().collect();
    let parse_pb = ProgressBar::new(files.len() as u64);
    parse_pb.set_style(
        ProgressStyle::with_template("[{bar:16.green}] {percent}% - {msg}")
            .unwrap()
            .progress_chars("██░")
    );

    let mut parsed_files = Vec::new();
    let mut chunk_count = 0;
    for file in files {
        // Chunks are labelled with the relative path, which is how the index and search results refer to files.
        let parsed = parse_source(Path::new(&file.relative_path), &file.content);
        if let Err(err) = &parsed {
            parse_pb.suspend(|| eprintln!("Warning: failed to parse {}: {}", file.relative_path, err));
        }
        let parse_failed = parsed.is_err();
        let chunks = parsed.unwrap_or_default();
        chunk_count += chunks.len();
        parsed_files.push(ParsedFile {
            relative_path: file.relative_path.clone(),
            content_hash: file.content_hash.clone(),
            chunks,
            parse_failed,
        });
        parse_pb.set_message(format!("{} chunks found", chunk_count));
        parse_pb.inc(1);
    }
    parse_pb.finish_with_message(format!("{} chunks found", chunk_count));
    parsed_files
}

// Each file is saved as soon as its chunks are embedded, so an interrupted run keeps the files it finished.
async fn embed_and_store(
    ollama: &Ollama,
    embedding_model: &str,
    conn: &mut rusqlite::Connection,
    files: Vec<ParsedFile>,
) -> Result<IndexOutcome, anyhow::Error> {
    let total_chunks: usize = files.iter().map(|file| file.chunks.len()).sum();
    let progress = ProgressBar::new(total_chunks as u64);
    progress.set_style(
        ProgressStyle::with_template("[{bar:16.green}] {percent}% - {pos}/{len} embeddings generated")
            .unwrap()
            .progress_chars("██░")
    );
    let mut embedder = ChunkEmbedder { ollama, embedding_model, progress, consecutive_failures: 0 };

    let mut outcome = IndexOutcome::default();
    for file in files {
        let embedded = embedder.embed(file.chunks).await.inspect_err(|_| embedder.progress.abandon())?;
        let is_complete = !file.parse_failed && embedded.skipped.is_empty();
        replace_file_chunks(conn, &FileUpdate {
            path: &file.relative_path,
            content_hash: is_complete.then_some(file.content_hash.as_str()),
            chunks: &embedded.chunks,
            embeddings: &embedded.embeddings,
        })
        .context("Failed to save chunks to database")?;

        outcome.stored_chunks += embedded.chunks.len();
        outcome.stored_files += 1;
        outcome.skipped_chunks.extend(embedded.skipped);
    }
    embedder.progress.finish();
    Ok(outcome)
}

impl ChunkEmbedder<'_> {
    // Chunks that fail on their own are skipped; a lost connection or a broken model stops the whole run.
    async fn embed(&mut self, chunks: Vec<CodeChunk>) -> Result<EmbeddedChunks, anyhow::Error> {
        let mut embedded = EmbeddedChunks::default();
        for chunk in chunks {
            let result = self.ollama.embed(self.embedding_model, &chunk.content).await;
            self.progress.inc(1);

            match result {
                Ok(embedding) => {
                    self.consecutive_failures = 0;
                    embedded.chunks.push(chunk);
                    embedded.embeddings.push(embedding);
                }
                Err(err) if is_connection_failure(&err) => {
                    anyhow::bail!(
                        "Lost connection to Ollama at {}. Files finished so far were saved; \
                         run `cbq index` again to continue.",
                        self.ollama.address()
                    );
                }
                Err(err) => {
                    self.consecutive_failures += 1;
                    if self.consecutive_failures == MAX_CONSECUTIVE_EMBEDDING_FAILURES {
                        return Err(err.context(format!(
                            "{} chunks in a row failed to embed. Files finished so far were saved; \
                             run `cbq index` again once the model works",
                            MAX_CONSECUTIVE_EMBEDDING_FAILURES
                        )));
                    }
                    embedded.skipped.push(format!(
                        "{}:{}-{}: {}",
                        chunk.file_path.display(),
                        chunk.start_line,
                        chunk.end_line,
                        err
                    ));
                }
            }
        }
        Ok(embedded)
    }
}

// Transport failures, including a server that dies mid-request; HTTP error statuses are chunk failures instead.
fn is_connection_failure(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<reqwest::Error>()
        .is_some_and(|error| error.is_connect() || error.is_timeout() || error.is_request())
}

fn print_skipped_chunks(skipped: &[String]) {
    if skipped.is_empty() {
        return;
    }
    crate::ui::formatter::print_warning_msg(&format!(
        "{} chunks could not be embedded and were skipped; their files will be retried next run:",
        skipped.len()
    ));
    for failure in skipped.iter().take(MAX_SKIPPED_ITEMS_LISTED) {
        println!("    {}", failure.dimmed());
    }
    if skipped.len() > MAX_SKIPPED_ITEMS_LISTED {
        println!("    ... and {} more", skipped.len() - MAX_SKIPPED_ITEMS_LISTED);
    }
    println!();
}

fn print_skipped_files(skipped: &[SkippedFile], root: &Path) {
    if skipped.is_empty() {
        return;
    }
    crate::ui::formatter::print_warning_msg(&format!("{} files were skipped:", skipped.len()));
    for file in skipped.iter().take(MAX_SKIPPED_ITEMS_LISTED) {
        let shown_path = file.path.strip_prefix(root).unwrap_or(&file.path);
        println!("    {} ({})", shown_path.display().to_string().dimmed(), file.reason.to_string().dimmed());
    }
    if skipped.len() > MAX_SKIPPED_ITEMS_LISTED {
        println!("    ... and {} more", skipped.len() - MAX_SKIPPED_ITEMS_LISTED);
    }
    if skipped.iter().any(|file| matches!(file.reason, SkipReason::TooLarge { .. })) {
        println!("    {}", "Raise the size limit with `cbq index --max-file-size <KB>`.".dimmed());
    }
    println!();
}

// Connects to Ollama and makes sure each model is downloaded, pulling any that are missing.
async fn prepare_ollama(config: &Config, models: &[&str]) -> Result<Ollama, anyhow::Error> {
    let ollama = Ollama::new(&config.ollama.host, config.ollama.port)?;
    ollama.check_running().await?;
    for model in models {
        ensure_model_available(&ollama, model).await?;
    }
    Ok(ollama)
}

async fn ensure_model_available(ollama: &Ollama, model: &str) -> Result<(), anyhow::Error> {
    if ollama.has_model(model).await? {
        return Ok(());
    }
    println!(
        "Model '{}' is not on {}. Pulling it now (this may take a while)...",
        model.cyan(),
        ollama.address()
    );

    let progress = ProgressBar::new(0);
    progress.set_style(
        ProgressStyle::with_template("{msg:30} [{bar:24.green}] {bytes}/{total_bytes}")
            .unwrap()
            .progress_chars("██░")
    );
    let pulled = ollama
        .pull_model(model, |update| {
            progress.set_message(update.status.clone());
            if let (Some(total), Some(completed)) = (update.total, update.completed) {
                progress.set_length(total);
                progress.set_position(completed);
            }
        })
        .await;
    progress.finish_and_clear();
    pulled?;

    println!("{} Model '{}' pulled", "✓".green(), model);
    Ok(())
}

fn open_project_index(directory: &Path, config: &Config) -> Result<(IndexedProject, rusqlite::Connection), anyhow::Error> {
    let Some(project) = find_indexed_project(directory)? else {
        return Err(missing_index_error(directory));
    };
    let conn = open_index(&project.db_path)?;
    ensure_index_model_matches(&conn, &config.ollama.embedding_model)?;
    Ok((project, conn))
}

fn missing_index_error(directory: &Path) -> anyhow::Error {
    let shown_directory = canonical_project_root(directory).unwrap_or_else(|_| directory.to_path_buf());
    let mut message = format!(
        "No index found for {} or any parent directory. Index the project first with: {}",
        shown_directory.display(),
        "cbq index <project-dir>".yellow().bold()
    );
    // Only a hint on top of the real error, so a failed lookup here is ignored.
    if let Ok(Some(legacy_dir)) = find_legacy_index(directory) {
        message.push_str(&format!(
            "\nNote: {} holds an index from an older cbq version, which is no longer used. \
             Re-index to replace it; that directory can then be deleted.",
            legacy_dir.display()
        ));
    }
    anyhow::anyhow!(message)
}

async fn run_search(query: &str, limit: Option<usize>, directory: &Path) {
    let config = match load_config() {
        Ok(c) => c,
        Err(_) => {
            let default_conf = crate::config::settings::Config::default();
            let _ = crate::config::settings::save_config(&default_conf);
            default_conf
        }
    };
    let limit = limit.unwrap_or(config.search.top_k);

    let (project, conn) = match open_project_index(directory, &config) {
        Ok(opened) => opened,
        Err(err) => {
            eprintln!("{} {}", "Error:".red().bold(), err);
            std::process::exit(1);
        }
    };

    let models = [config.ollama.embedding_model.as_str(), config.ollama.chat_model.as_str()];
    let ollama = match prepare_ollama(&config, &models).await {
        Ok(ollama) => ollama,
        Err(err) => {
            eprintln!("{} {:#}", "Error:".red().bold(), err);
            std::process::exit(1);
        }
    };

    println!(
        "Searching {} for: '{}'...",
        project.root.display().to_string().dimmed(),
        query.cyan()
    );

    let query_vector = match ollama.embed(&config.ollama.embedding_model, query).await {
        Ok(vec) => vec,
        Err(err) => {
            eprintln!("{} Failed to generate embedding for query: {}", "Error:".red().bold(), err);
            std::process::exit(1);
        }
    };

    match search_codebase(&conn, &query_vector, limit, config.search.similarity_threshold) {
        Ok(results) => {
            if results.is_empty() {
                println!("{}", "No relevant chunks found.".yellow());
                return;
            }
            print_search_results(&results);

            println!("{}", "🤖 [Ollama LLM Response]".blue().bold());
            let prompt = build_prompt(query, &results, &[]);
            if let Err(e) = stream_answer(&ollama, &config.ollama.chat_model, &prompt).await {
                eprintln!("\n{} Failed to get LLM response: {:#}", "Error:".red().bold(), e);
            }
            println!("\n");

            if let Err(e) = save_chat(query, &results) {
                eprintln!("Warning: Failed to save search history: {}", e);
            }
        }
        Err(err) => {
            eprintln!("{} Search failed: {}", "Error:".red().bold(), err);
            std::process::exit(1);
        }
    }
}

async fn stream_answer(ollama: &Ollama, chat_model: &str, prompt: &str) -> Result<String, anyhow::Error> {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    spinner.set_message("Thinking...".cyan().to_string());
    spinner.enable_steady_tick(Duration::from_millis(100));

    let mut answer = String::new();
    let stream_result = ollama.generate_stream(chat_model, prompt, |chunk| answer.push_str(chunk)).await;
    spinner.finish_and_clear();

    stream_result?;
    termimad::print_text(&answer);
    Ok(answer)
}

async fn run_chat_repl(directory: &Path) -> Result<(), anyhow::Error> {
    use std::io::{self, Write};

    let config = match load_config() {
        Ok(c) => c,
        Err(_) => {
            let default_conf = crate::config::settings::Config::default();
            let _ = crate::config::settings::save_config(&default_conf);
            default_conf
        }
    };

    let (project, conn) = open_project_index(directory, &config)?;

    println!("Checking Ollama connection...");
    let models = [config.ollama.embedding_model.as_str(), config.ollama.chat_model.as_str()];
    let ollama = prepare_ollama(&config, &models).await?;
    let session = Session { config: &config, ollama: &ollama, conn: &conn };

    println!("\n🤖 {}", format!("Codebase chat started for {}.", project.root.display()).cyan().bold());
    println!("Type 'exit' or 'quit' to end the session, or '/clear' to start a new conversation.");
    println!("Using embedding model: {}\n", config.ollama.embedding_model.yellow());

    let mut history: Vec<ChatTurn> = Vec::new();

    loop {
        print!("{} ", ">".green().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        let bytes_read = io::stdin().read_line(&mut input)?;
        if bytes_read == 0 {
            println!(); // Ctrl-D leaves the cursor on the prompt line
            break;
        }
        let question = input.trim();

        if question.is_empty() {
            continue;
        }
        if question == "exit" || question == "quit" {
            break;
        }
        if question == "/clear" {
            history.clear();
            println!("{}", "Conversation cleared.".cyan());
            continue;
        }

        match answer_chat_question(&session, &history, question).await {
            Ok(Some(answer)) => history.push(ChatTurn { question: question.to_string(), answer }),
            Ok(None) => {}
            Err(err) => eprintln!("{} {:#}", "Error:".red().bold(), err),
        }
    }

    println!("{}", "Exiting chat mode. Goodbye!".cyan());
    Ok(())
}

// Returns the answer so it can join the conversation, or None when there was nothing to answer from.
async fn answer_chat_question(
    session: &Session<'_>,
    history: &[ChatTurn],
    question: &str,
) -> Result<Option<String>, anyhow::Error> {
    let config = session.config;
    println!("Searching for matches...");
    let search_query = standalone_question(session, history, question).await;
    if search_query != question {
        println!("{} {}", "↳ searching for:".dimmed(), search_query.dimmed());
    }

    let query_vector = session
        .ollama
        .embed(&config.ollama.embedding_model, &search_query)
        .await
        .context("Failed to generate embedding")?;
    let results = search_codebase(session.conn, &query_vector, config.search.top_k, config.search.similarity_threshold)
        .context("Search failed")?;

    if results.is_empty() && history.is_empty() {
        println!("{}", "No relevant chunks found for this query.".yellow());
        return Ok(None);
    }
    if results.is_empty() {
        println!("{}", "No new code matched; answering from the conversation so far.".dimmed());
    } else {
        print_search_results(&results);
    }

    println!("{}", "🤖 [Ollama LLM Response]".blue().bold());
    let prompt = build_prompt(question, &results, history);
    let answer = stream_answer(session.ollama, &config.ollama.chat_model, &prompt)
        .await
        .context("Failed to get LLM response")?;
    println!();

    if let Err(e) = save_chat(question, &results) {
        eprintln!("Warning: Failed to save search history: {}", e);
    }
    Ok(Some(answer))
}

// Follow-ups like "what calls it?" retrieve poorly on their own, so they're rewritten using the conversation.
async fn standalone_question(session: &Session<'_>, history: &[ChatTurn], question: &str) -> String {
    if history.is_empty() {
        return question.to_string();
    }

    let prompt = build_rewrite_prompt(history, question);
    let rewritten = match session.ollama.generate_deterministic(&session.config.ollama.chat_model, &prompt).await {
        Ok(rewritten) => rewritten,
        Err(err) => {
            eprintln!("{} {}", "Warning: couldn't resolve the follow-up; searching for it as typed:".yellow(), err);
            return question.to_string();
        }
    };

    let first_line = rewritten.lines().map(str::trim).find(|line| !line.is_empty()).unwrap_or("");
    let first_line = first_line.trim_matches('"');
    if first_line.is_empty() {
        return question.to_string();
    }
    first_line.to_string()
}

fn print_search_results(results: &[SearchResult]) {
    println!("\n{} {} relevant chunks:\n", "🔍 Found".green(), results.len().to_string().yellow().bold());

    for (idx, result) in results.iter().enumerate() {
        let path_str = result.chunk.file_path.display().to_string();
        let is_test = path_str.contains("/test") || path_str.contains("test_") || path_str.starts_with("test");
        let badge = if is_test { "🧪" } else { "📄" };
        println!(
            "   {} {} {} {} [Score: {:.2}]",
            "└─".dimmed(),
            badge,
            (idx + 1).to_string().bold(),
            format!(
                "{}:{}-{}",
                path_str,
                result.chunk.start_line,
                result.chunk.end_line
            ).cyan(),
            result.score
        );
    }
    println!();
}

// Small local models resolve references far more reliably with worked examples than with instructions alone.
fn build_rewrite_prompt(history: &[ChatTurn], question: &str) -> String {
    format!(
        "Rewrite the user's latest question so it can be understood without the conversation, \
        for searching a codebase. Replace words like \"it\", \"its\", \"that\" or \"this\" with the \
        specific subject from the conversation. If the latest question is about a new subject, repeat it \
        unchanged. Reply with only the rewritten question, on one line.\n\n\
        Example 1:\n\
        User: How does the config loader read settings?\n\
        LATEST QUESTION: What happens if it fails?\n\
        REWRITTEN QUESTION: What happens if the config loader fails to read settings?\n\n\
        Example 2:\n\
        User: How does the config loader read settings?\n\
        LATEST QUESTION: How are search results ranked?\n\
        REWRITTEN QUESTION: How are search results ranked?\n\n\
        CONVERSATION:\n{}\n\
        LATEST QUESTION: {}\n\n\
        REWRITTEN QUESTION:",
        render_history(history),
        question
    )
}

fn render_history(history: &[ChatTurn]) -> String {
    let recent_turns = &history[history.len().saturating_sub(HISTORY_TURNS_IN_PROMPT)..];
    recent_turns
        .iter()
        .map(|turn| format!(
            "User: {}\nAssistant: {}\n",
            turn.question,
            shorten(&turn.answer, MAX_ANSWER_CHARS_IN_HISTORY)
        ))
        .collect()
}

fn shorten(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((cut, _)) => format!("{}…", &text[..cut]),
        None => text.to_string(),
    }
}

async fn run_analyze(directory: &Path, requested_source: Option<DiffSource>) -> Result<(), anyhow::Error> {
    let config = match load_config() {
        Ok(c) => c,
        Err(_) => crate::config::settings::Config::default(),
    };

    let diff_text = read_diff(directory, requested_source).await?;
    let files = parse_diff(&diff_text);
    if files.iter().all(|file| file.hunks.is_empty()) {
        println!("{}", "No changes to analyze.".yellow());
        return Ok(());
    }
    print_changed_files(&files);

    // Checked before pulling models, so a mismatched embedding model isn't downloaded only to be rejected.
    let index = match find_indexed_project(directory)? {
        Some(_) => Some(open_project_index(directory, &config)?),
        None => {
            println!(
                "{}",
                "No index found, so the review won't include related code. Run `cbq index <project-dir>` to add it."
                    .yellow()
            );
            None
        }
    };

    let mut models = vec![config.ollama.chat_model.as_str()];
    if index.is_some() {
        models.push(&config.ollama.embedding_model);
    }
    let ollama = prepare_ollama(&config, &models).await?;

    let related = match &index {
        Some((project, conn)) => {
            let session = Session { config: &config, ollama: &ollama, conn };
            find_related_code(&session, &files, &project.root).await?
        }
        None => Vec::new(),
    };
    if index.is_some() {
        print_related_code(&related);
    }

    let prompt = build_review_prompt(&files, &related);
    if prompt.omitted_hunks > 0 {
        println!(
            "{}",
            format!("{} hunks were left out of the review to fit the model's context.", prompt.omitted_hunks).dimmed()
        );
    }
    println!("{}", "🤖 [Ollama Review]".blue().bold());
    stream_answer(&ollama, &config.ollama.chat_model, &prompt.text)
        .await
        .context("Failed to get the review")?;
    println!();
    Ok(())
}

// A piped diff wins; otherwise git is asked for the requested changes, defaulting to everything uncommitted.
async fn read_diff(directory: &Path, requested_source: Option<DiffSource>) -> Result<String, anyhow::Error> {
    use std::io::{IsTerminal, Read};

    let source = match requested_source {
        Some(source) => source,
        None if !std::io::stdin().is_terminal() => {
            let mut diff = String::new();
            std::io::stdin().read_to_string(&mut diff)?;
            return Ok(diff);
        }
        None => DiffSource::Uncommitted,
    };
    println!("{}", format!("Reviewing {}", describe_diff_source(&source)).dimmed());
    read_git_diff(directory, &source).await
}

fn describe_diff_source(source: &DiffSource) -> String {
    match source {
        DiffSource::Uncommitted => "uncommitted changes (git diff HEAD)".to_string(),
        DiffSource::Staged => "staged changes (git diff --staged)".to_string(),
        DiffSource::SinceBase(base) => format!("changes since this branch left {} (git diff --merge-base {})", base, base),
    }
}

fn print_changed_files(files: &[FileDiff]) {
    println!("{} Changed files: {}", "→".cyan().bold(), files.len().to_string().yellow());
    for file in files {
        let (added, removed) = file.line_counts();
        println!(
            "  - {} ({}, {} {})",
            file.path.cyan(),
            describe_change(&file.change),
            format!("+{}", added).green(),
            format!("-{}", removed).red()
        );
    }
    println!();
}

// Searches with each changed hunk, dropping hits that are just the changed code itself.
async fn find_related_code(
    session: &Session<'_>,
    files: &[FileDiff],
    index_root: &Path,
) -> Result<Vec<SearchResult>, anyhow::Error> {
    let repository_root = find_repository_root(index_root).await;
    let hunks: Vec<(&FileDiff, &Hunk)> =
        files.iter().flat_map(|file| file.hunks.iter().map(move |hunk| (file, hunk))).collect();
    if hunks.len() > MAX_HUNKS_SEARCHED {
        println!(
            "{}",
            format!("Looking for related code around the first {} of {} hunks.", MAX_HUNKS_SEARCHED, hunks.len()).dimmed()
        );
    }

    let mut results = Vec::new();
    for (file, hunk) in hunks.into_iter().take(MAX_HUNKS_SEARCHED) {
        let changed_paths = changed_index_paths(file, repository_root.as_deref(), index_root);
        if changed_paths.is_empty() {
            continue; // the file lies outside the indexed directory
        }
        let query = format!("File: {}\n{}", file.path, hunk.text);
        let query_vector = session
            .ollama
            .embed(&session.config.ollama.embedding_model, &query)
            .await
            .context("Failed to embed a changed hunk")?;
        let hits = search_codebase(
            session.conn,
            &query_vector,
            RELATED_RESULTS_PER_HUNK,
            session.config.search.similarity_threshold,
        )
        .context("Search failed")?;
        results.extend(hits.into_iter().filter(|hit| {
            !changed_paths.iter().any(|path| is_changed_code(hit, path, hunk))
        }));
    }
    let mut related = merge_related_code(results, MAX_RELATED_CHUNKS_IN_REVIEW);
    for result in &mut related {
        result.chunk.file_path = to_diff_path(&result.chunk.file_path, repository_root.as_deref(), index_root);
    }
    Ok(related)
}

// A renamed file may still be indexed under its old path.
fn changed_index_paths(file: &FileDiff, repository_root: Option<&Path>, index_root: &Path) -> Vec<String> {
    let mut diff_paths = vec![file.path.as_str()];
    if let FileChange::Renamed { from } = &file.change {
        diff_paths.push(from);
    }
    diff_paths
        .into_iter()
        .filter_map(|path| to_index_path(path, repository_root, index_root))
        .collect()
}

fn print_related_code(related: &[SearchResult]) {
    if related.is_empty() {
        println!("{}\n", "No related code found in the index.".dimmed());
        return;
    }
    println!("{}", "Related code in the index:".underline().bold());
    for result in related {
        println!(
            "  - {} [Sim: {:.2}]",
            format!("{}:{}-{}", result.chunk.file_path.display(), result.chunk.start_line, result.chunk.end_line).cyan(),
            result.score
        );
    }
    println!();
}

fn build_prompt(query: &str, results: &[SearchResult], history: &[ChatTurn]) -> String {
    let mut context_str = String::new();
    for (i, result) in results.iter().enumerate() {
        let path_str = result.chunk.file_path.display().to_string();
        let is_test = path_str.contains("/test") || path_str.contains("test_") || path_str.starts_with("test");
        let label = if is_test { "(TEST FILE)" } else { "(SOURCE FILE)" };
        context_str.push_str(&format!(
            "--- Chunk {} {} ---\nFile: {}\nLines: {}-{}\n```\n{}\n```\n\n",
            i + 1,
            label,
            path_str,
            result.chunk.start_line,
            result.chunk.end_line,
            result.chunk.content
        ));
    }

    let conversation = if history.is_empty() {
        String::new()
    } else {
        format!(
            "CONVERSATION SO FAR (earlier questions and your answers; use it to understand follow-up questions):\n{}\n",
            render_history(history)
        )
    };

    format!(
        "You are an expert AI assistant that answers questions about a codebase.\n\
        You are given code chunks retrieved via semantic search, which may include both \
        source implementation files and test files.\n\n\
        IMPORTANT RULES:\n\
        - Prefer explaining from SOURCE FILE chunks over TEST FILE chunks.\n\
        - If only test files are available, say so explicitly and explain what the tests \
          REVEAL about the behavior, but clearly state that the actual implementation was \
          not retrieved.\n\
        - If the context is insufficient to answer the question accurately, say so. Do NOT \
          hallucinate or guess implementation details not visible in the provided code.\n\
        - Always cite the file name and line range when referring to a chunk.\n\n\
        CONTEXT:\n{}\n\n\
        {}QUESTION: {}\n\n\
        ANSWER:",
        context_str, conversation, query
    )
}

fn print_index_statistics(conn: &rusqlite::Connection, db_path: &Path) {
    match get_db_stats(conn) {
        Ok(stats) => {
            crate::ui::formatter::print_section("Database Index Statistics");
            println!("Location: {}", db_path.parent().unwrap().to_string_lossy().underline());
            
            let headers = vec!["Metric", "Value"];
            
            let mut lang_stmt = conn.prepare("SELECT DISTINCT file_path FROM chunks").unwrap();
            let paths_iter = lang_stmt.query_map([], |row| row.get::<_, String>(0)).unwrap();
            let mut languages = std::collections::HashSet::new();
            for p_res in paths_iter {
                if let Ok(p_str) = p_res {
                    let p = std::path::Path::new(&p_str);
                    if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                        languages.insert(ext.to_lowercase());
                    }
                }
            }
            let mut lang_vec: Vec<String> = languages.into_iter().collect();
            lang_vec.sort();
            
            let file_size = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);
            let file_size_mb = format!("{:.2} MB", file_size as f64 / 1024.0 / 1024.0);

            let rows = vec![
                vec!["Total Code Chunks".to_string(), stats.total_chunks.to_string()],
                vec!["Distinct Languages".to_string(), format!("{} ({})", lang_vec.len(), lang_vec.join(", "))],
                vec!["Disk Footprint Size".to_string(), file_size_mb],
            ];
            
            crate::ui::formatter::print_table(&headers, &rows);
        }
        Err(err) => {
            eprintln!("{} Failed to retrieve database statistics: {}", "Error:".red().bold(), err);
        }
    }
}
