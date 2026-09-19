pub mod cli;
pub mod services;
pub mod db;
pub mod config;
pub mod ui;

use std::path::{Path, PathBuf};
use std::time::Duration;
use anyhow::Context;
use clap::Parser;
use cli::args::{Cli, Commands};
use services::chunker::CodeChunk;
use services::file_discovery::discover_files;
use services::parser::parse_file;
use services::vector_search::{search_codebase, SearchResult};
use services::chat_history::{save_chat, get_history, export_history_to_markdown};
use services::ollama::{
    check_and_pull_model, check_ollama_status, generate_deterministic_response, generate_embedding,
    generate_response_stream,
};
use services::git::parse_diff;
use config::settings::{load_config, Config};
use db::schema::{init_db, open_index};
use db::location::{canonical_project_root, find_indexed_project, find_legacy_index, index_path_for, IndexedProject};
use db::index_metadata::ensure_index_model_matches;
use db::queries::{replace_index, get_db_stats};
use indicatif::{ProgressBar, ProgressStyle};
use colored::Colorize;

// A run of failures this long means Ollama or the model is broken, not individual chunks.
const MAX_CONSECUTIVE_EMBEDDING_FAILURES: usize = 10;
const MAX_SKIPPED_CHUNKS_LISTED: usize = 5;
const HISTORY_TURNS_IN_PROMPT: usize = 3;
// Long answers are cut so a few turns of history can't crowd the code context out of the prompt.
const MAX_ANSWER_CHARS_IN_HISTORY: usize = 1_500;

struct ChatTurn {
    question: String,
    answer: String,
}

#[derive(Default)]
struct EmbeddedChunks {
    chunks: Vec<CodeChunk>,
    embeddings: Vec<Vec<f32>>,
    skipped: Vec<String>,
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

            match discover_files(&path) {
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

                    println!("\n{}", "✓ Ready to index. Run: cbq index <path>".green());
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

            match discover_files(&path) {
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
        Some(Commands::Index { path }) => {
            if let Err(err) = run_index(&path).await {
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
        Some(Commands::Analyze { directory }) => {
            if let Err(err) = run_analyze(&directory).await {
                eprintln!("{} Analysis failed: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }
        }
        None => {
            if let Some(query) = args.default_query {
                run_search(&query, 5, Path::new(".")).await;
            } else {
                println!("No arguments provided. Run with --help to see usage.");
            }
        }
    }
}

async fn run_index(path: &Path) -> Result<(), anyhow::Error> {
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

    println!("Checking Ollama connection...");
    if !check_ollama_status(&config.ollama.host, config.ollama.port).await {
        anyhow::bail!("Ollama service is not running at {}:{}", config.ollama.host, config.ollama.port);
    }
    println!("{} Connected", "✓".green().bold());

    check_and_pull_model(&config.ollama.embedding_model).await?;
    println!("Embedding model: {}", config.ollama.embedding_model.yellow().bold());
    println!();

    let discovery = discover_files(&project_root).context("File discovery failed")?;
    let db_path = index_path_for(&project_root)?;
    println!(
        "Creating database... {}",
        db_path.to_string_lossy().underline().yellow()
    );
    let mut conn = init_db(&db_path).context("Failed to initialize database")?;

    println!("Parsing files...");
    let chunks = parse_project_files(&discovery.files, &project_root);
    println!();

    println!("Generating embeddings...");
    let embedded = embed_chunks(&config, chunks).await?;
    println!();
    print_skipped_chunks(&embedded.skipped);

    println!("Saving to database...");
    replace_index(&mut conn, &embedded.chunks, &embedded.embeddings, &config.ollama.embedding_model)
        .context("Failed to save chunks to database")?;
    println!("{} {} chunks stored\n", "✓".green().bold(), embedded.chunks.len().to_string().yellow().bold());

    print_index_statistics(&conn, &db_path);
    Ok(())
}

fn parse_project_files(files: &[PathBuf], project_root: &Path) -> Vec<CodeChunk> {
    let parse_pb = ProgressBar::new(files.len() as u64);
    parse_pb.set_style(
        ProgressStyle::with_template("[{bar:16.green}] {percent}% - {msg}")
            .unwrap()
            .progress_chars("██░")
    );

    let mut all_chunks = Vec::new();
    for file in files {
        match parse_file(file) {
            Ok(chunks) => {
                all_chunks.extend(chunks.into_iter().map(|chunk| relative_to_root(chunk, project_root)));
            }
            Err(err) => {
                eprintln!("\nWarning: failed to parse {}: {}", file.display(), err);
            }
        }
        parse_pb.set_message(format!("{} chunks found", all_chunks.len()));
        parse_pb.inc(1);
    }
    parse_pb.finish_with_message(format!("{} chunks found", all_chunks.len()));
    all_chunks
}

// Stored paths are relative to the project root so results read the same from any subdirectory.
fn relative_to_root(mut chunk: CodeChunk, project_root: &Path) -> CodeChunk {
    if let Ok(relative_path) = chunk.file_path.strip_prefix(project_root) {
        chunk.file_path = relative_path.to_path_buf();
    }
    chunk
}

// Chunks that fail on their own are skipped; a lost connection or a broken model stops the run
// before the existing index is touched.
async fn embed_chunks(config: &Config, chunks: Vec<CodeChunk>) -> Result<EmbeddedChunks, anyhow::Error> {
    let embed_pb = ProgressBar::new(chunks.len() as u64);
    embed_pb.set_style(
        ProgressStyle::with_template("[{bar:16.green}] {percent}% - {pos}/{len} embeddings generated")
            .unwrap()
            .progress_chars("██░")
    );

    let mut embedded = EmbeddedChunks::default();
    let mut consecutive_failures = 0;
    for chunk in chunks {
        let result = generate_embedding(
            &config.ollama.host,
            config.ollama.port,
            &config.ollama.embedding_model,
            &chunk.content,
        ).await;
        embed_pb.inc(1);

        match result {
            Ok(embedding) => {
                consecutive_failures = 0;
                embedded.chunks.push(chunk);
                embedded.embeddings.push(embedding);
            }
            Err(err) if is_connection_failure(&err) => {
                embed_pb.abandon();
                anyhow::bail!(
                    "Lost connection to Ollama at {}:{}; the existing index was left unchanged",
                    config.ollama.host,
                    config.ollama.port
                );
            }
            Err(err) => {
                consecutive_failures += 1;
                if consecutive_failures == MAX_CONSECUTIVE_EMBEDDING_FAILURES {
                    embed_pb.abandon();
                    return Err(err.context(format!(
                        "{} chunks in a row failed to embed; the existing index was left unchanged",
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
    embed_pb.finish();

    if embedded.chunks.is_empty() && !embedded.skipped.is_empty() {
        anyhow::bail!("No chunks could be embedded; the existing index was left unchanged");
    }
    Ok(embedded)
}

fn is_connection_failure(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<reqwest::Error>()
        .is_some_and(|error| error.is_connect() || error.is_timeout())
}

fn print_skipped_chunks(skipped: &[String]) {
    if skipped.is_empty() {
        return;
    }
    crate::ui::formatter::print_warning_msg(&format!(
        "{} chunks could not be embedded and were skipped:",
        skipped.len()
    ));
    for failure in skipped.iter().take(MAX_SKIPPED_CHUNKS_LISTED) {
        println!("    {}", failure.dimmed());
    }
    if skipped.len() > MAX_SKIPPED_CHUNKS_LISTED {
        println!("    ... and {} more", skipped.len() - MAX_SKIPPED_CHUNKS_LISTED);
    }
    println!();
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

async fn run_search(query: &str, limit: usize, directory: &Path) {
    let config = match load_config() {
        Ok(c) => c,
        Err(_) => {
            let default_conf = crate::config::settings::Config::default();
            let _ = crate::config::settings::save_config(&default_conf);
            default_conf
        }
    };

    let (project, conn) = match open_project_index(directory, &config) {
        Ok(opened) => opened,
        Err(err) => {
            eprintln!("{} {}", "Error:".red().bold(), err);
            std::process::exit(1);
        }
    };

    if !check_ollama_status(&config.ollama.host, config.ollama.port).await {
        eprintln!(
            "{} Ollama service is not running at {}:{}",
            "Error:".red().bold(),
            config.ollama.host,
            config.ollama.port
        );
        std::process::exit(1);
    }
    
    if let Err(e) = check_and_pull_model(&config.ollama.embedding_model).await {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }
    if let Err(e) = check_and_pull_model(&config.ollama.chat_model).await {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

    println!(
        "Searching {} for: '{}'...",
        project.root.display().to_string().dimmed(),
        query.cyan()
    );

    let query_vector = match generate_embedding(
        &config.ollama.host,
        config.ollama.port,
        &config.ollama.embedding_model,
        query,
    ).await {
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
            if let Err(e) = stream_answer(&config, &prompt).await {
                eprintln!("\n{} Failed to get LLM response: {}", "Error:".red().bold(), e);
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

async fn stream_answer(config: &Config, prompt: &str) -> Result<String, anyhow::Error> {
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
    let stream_result = generate_response_stream(
        &config.ollama.host,
        config.ollama.port,
        &config.ollama.chat_model,
        prompt,
        |chunk| {
            answer.push_str(chunk);
        }
    ).await;
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
    if !check_ollama_status(&config.ollama.host, config.ollama.port).await {
        eprintln!(
            "{} Ollama service is not running at {}:{}",
            "Error:".red().bold(),
            config.ollama.host,
            config.ollama.port
        );
        std::process::exit(1);
    }
    
    if let Err(e) = check_and_pull_model(&config.ollama.embedding_model).await {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }
    if let Err(e) = check_and_pull_model(&config.ollama.chat_model).await {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

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

        match answer_chat_question(&config, &conn, &history, question).await {
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
    config: &Config,
    conn: &rusqlite::Connection,
    history: &[ChatTurn],
    question: &str,
) -> Result<Option<String>, anyhow::Error> {
    println!("Searching for matches...");
    let search_query = standalone_question(config, history, question).await;
    if search_query != question {
        println!("{} {}", "↳ searching for:".dimmed(), search_query.dimmed());
    }

    let query_vector = generate_embedding(
        &config.ollama.host,
        config.ollama.port,
        &config.ollama.embedding_model,
        &search_query,
    ).await.context("Failed to generate embedding")?;
    let results = search_codebase(conn, &query_vector, config.search.top_k, config.search.similarity_threshold)
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
    let answer = stream_answer(config, &prompt).await.context("Failed to get LLM response")?;
    println!();

    if let Err(e) = save_chat(question, &results) {
        eprintln!("Warning: Failed to save search history: {}", e);
    }
    Ok(Some(answer))
}

// Follow-ups like "what calls it?" retrieve poorly on their own, so they're rewritten using the conversation.
async fn standalone_question(config: &Config, history: &[ChatTurn], question: &str) -> String {
    if history.is_empty() {
        return question.to_string();
    }

    let rewritten = match generate_deterministic_response(
        &config.ollama.host,
        config.ollama.port,
        &config.ollama.chat_model,
        &build_rewrite_prompt(history, question),
    ).await {
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

async fn run_analyze(directory: &Path) -> Result<(), anyhow::Error> {
    use std::io::{self, Read};

    let mut diff_input = String::new();
    io::stdin().read_to_string(&mut diff_input)?;

    if diff_input.trim().is_empty() {
        println!("{}", "No diff provided. Usage: git diff | cbq analyze".yellow());
        return Ok(());
    }

    println!("Analyzing changes...\n");
    let parsed_diffs = parse_diff(&diff_input);
    
    if parsed_diffs.is_empty() {
        println!("No significant code modifications found in the diff.");
        return Ok(());
    }

    println!("{} Modified files: {}", "→".cyan().bold(), parsed_diffs.len().to_string().yellow());
    for diff in &parsed_diffs {
        println!("  - {} ({} lines added/modified)", diff.file_path.cyan(), diff.added_lines.len());
    }
    println!();

    let config = match load_config() {
        Ok(c) => c,
        Err(_) => crate::config::settings::Config::default(),
    };

    if find_indexed_project(directory)?.is_none() {
        println!("{}", "No index found. Run `cbq index <project-dir>` to enable impact analysis.".yellow());
        return Ok(());
    }
    // Checked before pulling the model, so a mismatched model isn't downloaded only to be rejected.
    let (_, conn) = open_project_index(directory, &config)?;

    if let Err(e) = check_and_pull_model(&config.ollama.embedding_model).await {
        eprintln!("{} {}", "Error:".red().bold(), e);
        std::process::exit(1);
    }

    println!("{}", "Semantic Impact Analysis:".underline().bold());
    for diff in &parsed_diffs {
        if diff.added_lines.is_empty() {
            continue;
        }

        let added_code = diff.added_lines.join("\n");
        let query_vector = match generate_embedding(
            &config.ollama.host,
            config.ollama.port,
            &config.ollama.embedding_model,
            &added_code,
        ).await {
            Ok(vec) => vec,
            Err(_) => continue,
        };

        match search_codebase(&conn, &query_vector, 2, config.search.similarity_threshold) {
            Ok(results) => {
                if results.is_empty() {
                    println!("  {} No closely related files found in the index for {}", "○".dimmed(), diff.file_path.cyan());
                } else {
                    println!("  {} Related context for {}:", "✓".green(), diff.file_path.cyan());
                    for result in results {
                        println!(
                            "      - {} (Line {} to {}) [Sim: {:.2}]",
                            result.chunk.file_path.display(),
                            result.chunk.start_line,
                            result.chunk.end_line,
                            result.score
                        );
                    }
                }
            }
            Err(err) => return Err(err.context("Search failed")),
        }
    }
    println!();
    
    println!("{}", "Suggestions:".underline().bold());
    println!("  - Review the related context above to ensure APIs are updated symmetrically.");
    println!("  - Consider running `cargo test` to ensure changes do not break existing logic.");

    Ok(())
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
