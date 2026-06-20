pub mod cli;

use clap::Parser;
use cli::args::{Cli, Commands};

fn main() {
    let args = Cli::parse();
    match args.command {
        Some(Commands::Init { path }) => {
            println!("Initializing... path: {}", path.display());
        }
        None => {
            if let Some(query) = args.default_query {
                println!("Feature not yet implemented: {}", query);
            } else {
                println!("No arguments provided. Run --help to see usage.");
            }
        }
    }
}