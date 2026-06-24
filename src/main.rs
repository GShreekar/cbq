pub mod cli;
pub mod services;
pub mod db;
pub mod config;
pub mod ui;

use std::time::Duration;
use clap::Parser;
use cli::args::{Cli, Commands};
use services::file_discovery::discover_files;
use services::parser::parse_file;
use services::vector_search::search_codebase;
use services::chat_history::{save_chat, get_history, export_history_to_markdown};
use services::ollama::{check_ollama_status, generate_embedding};
use config::settings::load_config;
use db::schema::{get_db_path, init_db};
use db::queries::{clear_chunks, insert_chunks, get_db_stats};
use indicatif::{ProgressBar, ProgressStyle};
use colored::Colorize;

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

                    println!("\n{}", "✓ Ready to index. Run: cargo run -- index <path>".green());
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
            println!("Indexing {}...", path.display().to_string().cyan());

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
                eprintln!(
                    "{} Ollama service is not running at {}:{}",
                    "Error:".red().bold(),
                    config.ollama.host,
                    config.ollama.port
                );
                std::process::exit(1);
            }
            println!("{} Connected", "✓".green().bold());
            println!("Embedding model: {}", config.ollama.embedding_model.yellow().bold());
            println!();

            let discovery = match discover_files(&path) {
                Ok(res) => res,
                Err(err) => {
                    eprintln!("{} {}", "Error during file discovery:".red().bold(), err);
                    std::process::exit(1);
                }
            };

            let db_path = match get_db_path(&path) {
                Ok(p) => p,
                Err(err) => {
                    eprintln!("{} {}", "Failed to determine database path:".red().bold(), err);
                    std::process::exit(1);
                }
            };

            println!(
                "Creating database... {}",
                db_path.to_string_lossy().underline().yellow()
            );

            let mut conn = match init_db(&db_path) {
                Ok(c) => c,
                Err(err) => {
                    eprintln!("{} {}", "Failed to initialize database:".red().bold(), err);
                    std::process::exit(1);
                }
            };

            println!("Parsing files...");
            let parse_pb = ProgressBar::new(discovery.files.len() as u64);
            parse_pb.set_style(
                ProgressStyle::with_template("[{bar:16.green}] {percent}% - {msg}")
                    .unwrap()
                    .progress_chars("██░")
            );

            let mut all_chunks = Vec::new();
            for file in &discovery.files {
                match parse_file(file) {
                    Ok(chunks) => {
                        all_chunks.extend(chunks);
                    }
                    Err(err) => {
                        eprintln!("\nWarning: failed to parse {}: {}", file.display(), err);
                    }
                }
                parse_pb.set_message(format!("{} chunks found", all_chunks.len()));
                parse_pb.inc(1);
            }
            parse_pb.finish_with_message(format!("{} chunks found", all_chunks.len()));
            println!();

            println!("Generating embeddings...");
            let embed_pb = ProgressBar::new(all_chunks.len() as u64);
            embed_pb.set_style(
                ProgressStyle::with_template("[{bar:16.green}] {percent}% - {pos}/{len} embeddings generated")
                    .unwrap()
                    .progress_chars("██░")
            );

            let mut embeddings = Vec::new();
            for chunk in &all_chunks {
                match generate_embedding(
                    &config.ollama.host,
                    config.ollama.port,
                    &config.ollama.embedding_model,
                    &chunk.content,
                ).await {
                    Ok(emb) => {
                        embeddings.push(emb);
                    }
                    Err(err) => {
                        embed_pb.finish_and_clear();
                        eprintln!(
                            "\n{} Failed to generate embedding for chunk in {}: {}",
                            "Error:".red().bold(),
                            chunk.file_path.display(),
                            err
                        );
                        std::process::exit(1);
                    }
                }
                embed_pb.inc(1);
            }
            embed_pb.finish_with_message(format!("{} embeddings generated", embeddings.len()));
            println!();

            println!("Saving to database...");
            if let Err(err) = clear_chunks(&conn) {
                eprintln!("{} Failed to clear old database chunks: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }

            if let Err(err) = insert_chunks(&mut conn, &all_chunks, &embeddings) {
                eprintln!("{} Failed to save chunks to database: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }

            println!("{} {} chunks stored\n", "✓".green().bold(), all_chunks.len().to_string().yellow().bold());

            match get_db_stats(&conn) {
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
                    
                    let file_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
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
        Some(Commands::Search { query, limit }) => {
            run_search(&query, limit).await;
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
                        "ollama.embedding_model" => conf.ollama.embedding_model = value,
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
        Some(Commands::Chat) => {
            if let Err(err) = run_chat_repl().await {
                eprintln!("{} Chat session error: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }
        }
        None => {
            if let Some(query) = args.default_query {
                run_search(&query, 5).await;
            } else {
                println!("No arguments provided. Run with --help to see usage.");
            }
        }
    }
}

async fn run_search(query: &str, limit: usize) {
    let config = match load_config() {
        Ok(c) => c,
        Err(_) => {
            let default_conf = crate::config::settings::Config::default();
            let _ = crate::config::settings::save_config(&default_conf);
            default_conf
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

    let project_path = std::path::Path::new(".");
    let db_path = match get_db_path(project_path) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("{} {}", "Database path error:".red().bold(), err);
            std::process::exit(1);
        }
    };

    if !db_path.exists() {
        eprintln!(
            "{} Database does not exist. Please index the workspace first using: {}",
            "Error:".red().bold(),
            "cargo run -- index .".yellow().bold()
        );
        std::process::exit(1);
    }

    let conn = match rusqlite::Connection::open(&db_path) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("{} Failed to open connection: {}", "Error:".red().bold(), err);
            std::process::exit(1);
        }
    };

    println!("Searching database for: '{}'...", query.cyan());

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

            println!("\nFound {} relevant chunks:\n", results.len().to_string().yellow().bold());

            for (idx, result) in results.iter().enumerate() {
                println!(
                    "{}. {} [Score: {:.2}]",
                    (idx + 1).to_string().bold(),
                    format!(
                        "{}:{}-{}",
                        result.chunk.file_path.display(),
                        result.chunk.start_line,
                        result.chunk.end_line
                    ).magenta().underline(),
                    result.score
                );

                for line in result.chunk.content.lines().take(5) {
                    let highlighted = crate::ui::formatter::highlight_code(line);
                    println!("   {}", highlighted);
                }
                if result.chunk.content.lines().count() > 5 {
                    println!("   {}", "...".dimmed());
                }
                println!();
            }

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

async fn run_chat_repl() -> Result<(), anyhow::Error> {
    use std::io::{self, Write};

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
        eprintln!(
            "{} Ollama service is not running at {}:{}",
            "Error:".red().bold(),
            config.ollama.host,
            config.ollama.port
        );
        std::process::exit(1);
    }

    let project_path = std::path::Path::new(".");
    let db_path = match get_db_path(project_path) {
        Ok(p) => p,
        Err(err) => return Err(anyhow::anyhow!("Database path error: {}", err)),
    };

    let conn = rusqlite::Connection::open(&db_path)?;

    println!("\n🤖 {}", "Codebase chat started. Type 'exit' or 'quit' to end session.".cyan().bold());
    println!("Using embedding model: {}\n", config.ollama.embedding_model.yellow());

    let mut context_queries = Vec::new();

    loop {
        print!("{} ", ">".green().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let query = input.trim();

        if query.is_empty() {
            continue;
        }

        if query == "exit" || query == "quit" {
            println!("{}", "Exiting chat mode. Goodbye!".cyan());
            break;
        }

        context_queries.push(query.to_string());
        println!("Searching for matches...");

        let query_vector = match generate_embedding(
            &config.ollama.host,
            config.ollama.port,
            &config.ollama.embedding_model,
            query,
        ).await {
            Ok(vec) => vec,
            Err(err) => {
                eprintln!("{} Failed to generate embedding: {}", "Error:".red().bold(), err);
                continue;
            }
        };

        match search_codebase(&conn, &query_vector, 3, config.search.similarity_threshold) {
            Ok(results) => {
                if results.is_empty() {
                    println!("{}", "No relevant chunks found for this query.".yellow());
                    continue;
                }

                println!("\nFound {} relevant chunks:\n", results.len().to_string().yellow().bold());

                for (idx, result) in results.iter().enumerate() {
                    println!(
                        "   {}. {} [Score: {:.2}]",
                        (idx + 1).to_string().bold(),
                        format!(
                            "{}:{}-{}",
                            result.chunk.file_path.display(),
                            result.chunk.start_line,
                            result.chunk.end_line
                        ).magenta().underline(),
                        result.score
                    );

                    for line in result.chunk.content.lines().take(3) {
                        let highlighted = crate::ui::formatter::highlight_code(line);
                        println!("      {}", highlighted);
                    }
                    if result.chunk.content.lines().count() > 3 {
                        println!("      {}", "...".dimmed());
                    }
                    println!();
                }

                if let Some(top_match) = results.first() {
                    println!("{}", "🤖 [Mock LLM Response]".blue().bold());
                    println!(
                        "Based on the context found in {} ({} {} named '{}'):",
                        top_match.chunk.file_path.display().to_string().yellow(),
                        "logical".dimmed(),
                        top_match.chunk.chunk_type.magenta(),
                        top_match.chunk.name.bold()
                    );
                    println!(
                        "   We found the implementation details starting at line {}. This snippet addresses your query about '{}'.",
                        top_match.chunk.start_line,
                        query.bold()
                    );
                    println!();
                }

                if let Err(e) = save_chat(query, &results) {
                    eprintln!("Warning: Failed to save search history: {}", e);
                }
            }
            Err(err) => {
                eprintln!("{} Search failed: {}", "Error:".red().bold(), err);
            }
        }
    }

    Ok(())
}