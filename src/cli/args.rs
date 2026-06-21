use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "cbq",
    version = "0.1.0",
    about = "Local-first codebase semantic Q&A and analysis CLI tool",
    after_help = "If no subcommand is provided, the query argument will run a default semantic search."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    #[arg(
        help = "The question or query to run against the indexed codebase (runs if no subcommand is specified)",
        required = false
    )]
    pub default_query: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    #[command(about = "Initialize and index code files in a target directory")]
    Init {
        #[arg(help = "The target directory path to index", default_value = "./")]
        path: PathBuf,
    },
    #[command(about = "Parse code files in a directory and print detected chunks")]
    Parse {
        #[arg(help = "The directory path to scan and parse", default_value = "./")]
        path: PathBuf,
    },
}
