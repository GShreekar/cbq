pub mod cli;
pub mod services;
pub mod db;

use std::time::Duration;
use clap::Parser;
use cli::args::{Cli, Commands};
use services::file_discovery::discover_files;
use services::parser::parse_file;
use db::schema::{get_db_path, init_db};
use db::queries::{clear_chunks, insert_chunks, get_db_stats};
use indicatif::{ProgressBar, ProgressStyle};
use colored::Colorize;

fn main() {
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

            let discovery = match discover_files(&path) {
                Ok(res) => res,
                Err(err) => {
                    eprintln!("{} {}", "Error during file discovery:".red().bold(), err);
                    std::process::exit(1);
                }
            };

            let file_pb = ProgressBar::new(discovery.files.len() as u64);
            file_pb.set_style(
                ProgressStyle::with_template("[{bar:16.green}] {percent}% - {pos} files processed")
                    .unwrap()
                    .progress_chars("██░")
            );
            for _ in 0..discovery.files.len() {
                file_pb.inc(1);
            }
            file_pb.finish();
            println!();

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

            println!("Saving to database...");
            if let Err(err) = clear_chunks(&conn) {
                eprintln!("{} Failed to clear old database chunks: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }

            if let Err(err) = insert_chunks(&mut conn, &all_chunks) {
                eprintln!("{} Failed to save chunks to database: {}", "Error:".red().bold(), err);
                std::process::exit(1);
            }

            println!("{} {} chunks stored\n", "✓".green().bold(), all_chunks.len().to_string().yellow().bold());

            match get_db_stats(&conn) {
                Ok(stats) => {
                    println!("Database: {}", db_path.parent().unwrap().to_string_lossy().underline());
                    println!("  - Chunks: {}", stats.total_chunks.to_string().bold());
                    
                    let mut lang_stmt = match conn.prepare("SELECT DISTINCT file_path FROM chunks") {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("Error querying languages: {}", e);
                            return;
                        }
                    };
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
                    let lang_list = lang_vec.join(", ");
                    println!("  - Languages: {} ({})", lang_vec.len(), lang_list.blue());

                    let file_size = std::fs::metadata(&db_path)
                        .map(|m| m.len())
                        .unwrap_or(0);
                    let file_size_mb = file_size as f64 / 1024.0 / 1024.0;
                    println!("  - Size: {:.2} MB", file_size_mb);
                }
                Err(err) => {
                    eprintln!("{} Failed to retrieve database statistics: {}", "Error:".red().bold(), err);
                }
            }
        }
        None => {
            if let Some(query) = args.default_query {
                print!("Feature not yet implemented: {}", query);
            } else {
                println!("No arguments provided. Run with --help to see usage.");
            }
        }
    }
}