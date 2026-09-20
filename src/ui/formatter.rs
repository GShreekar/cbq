use colored::Colorize;
use crate::ui::colors::{paint_error, paint_primary, paint_success, paint_warning};

pub fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    if headers.is_empty() || rows.is_empty() {
        return;
    }

    let mut col_widths: Vec<usize> = headers.iter().map(|header| header.chars().count()).collect();

    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < col_widths.len() {
                col_widths[i] = col_widths[i].max(cell.chars().count());
            }
        }
    }

    let separator = col_widths
        .iter()
        .map(|w| "─".repeat(*w + 2))
        .collect::<Vec<String>>()
        .join("┼");
    
    println!("┌{}┐", separator.replace('┼', "┬"));

    let header_line = headers
        .iter()
        .enumerate()
        .map(|(i, header)| format!(" {} ", pad(header, col_widths[i]).bold().cyan()))
        .collect::<Vec<String>>()
        .join("│");
    println!("│{}│", header_line);

    println!("├{}┤", separator);

    for row in rows {
        let row_line = row
            .iter()
            .enumerate()
            .map(|(i, cell)| format!(" {} ", pad(cell, col_widths[i])))
            .collect::<Vec<String>>()
            .join("│");
        println!("│{}│", row_line);
    }

    println!("└{}┘", separator.replace('┼', "┴"));
}

fn pad(text: &str, width: usize) -> String {
    let padding = width.saturating_sub(text.chars().count());
    format!("{}{}", text, " ".repeat(padding))
}

pub fn print_section(title: &str) {
    println!("\n{}", paint_primary(title));
}

pub fn print_success_msg(msg: &str) {
    println!("{} {}", "✓".green().bold(), paint_success(msg));
}

/// Prints a warning to stderr, where it stays out of piped output.
pub fn print_warning_msg(msg: &str) {
    eprintln!("{} {}", "⚠".yellow().bold(), paint_warning(msg));
}

/// Prints an error to stderr, where it stays out of piped output.
pub fn print_error(msg: &str) {
    eprintln!("{} {}", paint_error("Error:"), msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_counts_characters_not_bytes() {
        assert_eq!(pad("café", 6), "café  ");
    }

    #[test]
    fn text_at_the_column_width_is_left_alone() {
        assert_eq!(pad("rust", 4), "rust");
    }

    #[test]
    fn text_wider_than_the_column_is_not_truncated() {
        assert_eq!(pad("markdown", 4), "markdown");
    }
}
