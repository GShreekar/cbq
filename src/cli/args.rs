use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "cbq",
    version = env!("CARGO_PKG_VERSION"),
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
    #[command(about = "Report which files in a directory would be indexed")]
    Init {
        #[arg(help = "The target directory path to scan", default_value = "./")]
        path: PathBuf,
    },
    #[command(about = "Parse code files in a directory and print detected chunks")]
    Parse {
        #[arg(help = "The directory path to scan and parse", default_value = "./")]
        path: PathBuf,
    },
    #[command(about = "Index code files in a directory, re-embedding only files that changed since the last run")]
    Index {
        #[arg(help = "The target directory path to index", default_value = "./")]
        path: PathBuf,

        #[arg(long, help = "Rebuild the whole index instead of updating only changed files")]
        force: bool,

        #[arg(
            long = "max-file-size",
            value_name = "KB",
            help = "Skip files larger than this many kilobytes",
            default_value_t = 512
        )]
        max_file_size_kb: u64,
    },
    #[command(about = "Search the indexed codebase using keywords")]
    Search {
        #[arg(help = "The question or term to search for")]
        query: String,

        #[arg(help = "The maximum number of results to display [default: search.top_k from config]", short, long)]
        limit: Option<usize>,

        #[arg(
            short = 'C',
            long,
            help = "Project directory to use instead of the current one",
            default_value = "."
        )]
        directory: PathBuf,
    },
    #[command(about = "Manage global configuration settings")]
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    #[command(about = "Display the questions asked about this project")]
    History {
        #[arg(help = "How many recent questions to show", short, long, default_value_t = 20)]
        limit: usize,

        #[arg(
            short = 'C',
            long,
            help = "Project directory to use instead of the current one",
            default_value = "."
        )]
        directory: PathBuf,
    },
    #[command(about = "Export this project's questions and answers to a Markdown file")]
    Export {
        #[arg(help = "Where to write the transcript", short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        #[arg(
            short = 'C',
            long,
            help = "Project directory to use instead of the current one",
            default_value = "."
        )]
        directory: PathBuf,
    },
    #[command(about = "Start an interactive chat session to query the codebase")]
    Chat {
        #[arg(
            short = 'C',
            long,
            help = "Project directory to use instead of the current one",
            default_value = "."
        )]
        directory: PathBuf,
    },
    #[command(
        about = "Review a git diff for bugs and affected code",
        long_about = "Review a git diff for bugs and affected code.\n\n\
            Reads a diff piped on stdin (git diff | cbq analyze). With nothing piped, runs git itself: \
            all uncommitted changes by default, or the changes selected by --staged or --base."
    )]
    Analyze {
        #[arg(long, help = "Review only staged changes", conflicts_with = "base")]
        staged: bool,

        #[arg(
            long,
            value_name = "REF",
            help = "Review everything since this branch diverged from REF, including uncommitted work"
        )]
        base: Option<String>,

        #[arg(
            short = 'C',
            long,
            help = "Project directory to use instead of the current one",
            default_value = "."
        )]
        directory: PathBuf,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    #[command(about = "Initialize default configuration file")]
    Init,

    #[command(about = "Display current configuration parameters")]
    Get,

    #[command(about = "Set a configuration parameter (e.g. search.top_k 10)")]
    Set {
        #[arg(help = "The config key path to set (e.g. search.top_k, ollama.host")]
        key: String,

        #[arg(help = "The value to set")]
        value: String,
    },
}