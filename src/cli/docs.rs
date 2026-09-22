use clap::Command;

/// Renders a man page from the command definitions, so it can't drift from `--help`.
pub fn render_man_page(command: &Command) -> String {
    let name = command.get_name().to_uppercase();
    let version = command.get_version().unwrap_or("");
    let about = command.get_about().map(|about| about.to_string()).unwrap_or_default();

    let mut page = format!(
        ".TH {} 1 \"\" \"{} {}\" \"User Commands\"\n.SH NAME\n{} \\- {}\n.SH SYNOPSIS\n.B {}\n\
         [\\fIOPTIONS\\fR] <\\fICOMMAND\\fR> [\\fIARGS\\fR]\n.SH DESCRIPTION\n{}\n",
        name,
        command.get_name(),
        version,
        command.get_name(),
        escape(&about),
        command.get_name(),
        escape(&about)
    );

    page.push_str(".SH COMMANDS\n");
    for subcommand in command.get_subcommands().filter(|sub| !sub.is_hide_set()) {
        page.push_str(&format!(
            ".TP\n.B {}\n{}\n",
            escape(subcommand.get_name()),
            escape(&subcommand.get_about().map(|about| about.to_string()).unwrap_or_default())
        ));
    }

    page.push_str(".SH OPTIONS\n");
    for option in command.get_arguments().filter(|argument| !argument.is_hide_set()) {
        page.push_str(&format!(
            ".TP\n.B {}\n{}\n",
            escape(&describe_flags(option)),
            escape(&option.get_help().map(|help| help.to_string()).unwrap_or_default())
        ));
    }

    page.push_str(
        ".SH FILES\n.TP\n.B ~/.cbq/config.toml\nGlobal settings.\n\
         .TP\n.B .cbq.toml\nPer-project settings, which override the global ones except for the Ollama address.\n\
         .TP\n.B ~/.cbq/codebases/\nOne index per project, holding its code and embeddings.\n\
         .TP\n.B .cbqignore\nPaths to keep out of the index, in .gitignore syntax.\n",
    );
    page.push_str(".SH EXIT STATUS\n.TP\n.B 0\nThe command succeeded.\n.TP\n.B 1\nThe command failed, or `cbq doctor` found a problem.\n");
    page
}

fn describe_flags(argument: &clap::Arg) -> String {
    let mut flags = Vec::new();
    if let Some(short) = argument.get_short() {
        flags.push(format!("-{}", short));
    }
    if let Some(long) = argument.get_long() {
        flags.push(format!("--{}", long));
    }
    if flags.is_empty() {
        flags.push(argument.get_id().to_string());
    }
    flags.join(", ")
}

// A leading dot or apostrophe starts a roff request, and a backslash starts an escape.
fn escape(text: &str) -> String {
    let escaped = text.replace('\\', "\\\\").replace('-', "\\-");
    match escaped.starts_with('.') || escaped.starts_with('\'') {
        true => format!("\\&{}", escaped),
        false => escaped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    use crate::cli::args::Cli;

    fn page() -> String {
        render_man_page(&Cli::command())
    }

    #[test]
    fn the_page_opens_with_a_title_and_name() {
        let page = page();
        assert!(page.starts_with(".TH CBQ 1"));
        assert!(page.contains(".SH NAME\ncbq \\- "));
    }

    #[test]
    fn every_section_a_reader_expects_is_present() {
        let page = page();
        for section in [".SH SYNOPSIS", ".SH DESCRIPTION", ".SH COMMANDS", ".SH OPTIONS", ".SH FILES", ".SH EXIT STATUS"] {
            assert!(page.contains(section), "missing {}", section);
        }
    }

    #[test]
    fn the_commands_section_lists_the_subcommands() {
        let page = page();
        assert!(page.contains(".B index"));
        assert!(page.contains(".B search"));
        assert!(page.contains(".B doctor"));
    }

    #[test]
    fn a_leading_dot_is_kept_from_starting_a_roff_request() {
        assert_eq!(escape(".cbq.toml"), "\\&.cbq.toml");
    }

    #[test]
    fn hyphens_are_escaped_so_they_stay_hyphens() {
        assert_eq!(escape("--json"), "\\-\\-json");
    }
}
