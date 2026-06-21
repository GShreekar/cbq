pub mod cli;
pub mod services;

use std::time::Duration;
use clap::Parser;
use cli::args::{Cli, Commands};
use services::file_discovery::discover_files;
use services::parser::parse_file;
use indicatif::ProgressBar;
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

                    println!("\n{}", "✓ Ready to index. Run: cbq index-start".green());
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
        None => {
            if let Some(query) = args.default_query {
                print!("Feature not yet implemented: {}", query);
            } else {
                println!("No arguments provided. Run with --help to see usage.");
            }
        }
    }
}