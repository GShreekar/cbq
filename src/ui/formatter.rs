use colored::Colorize;
use crate::ui::colors::{paint_muted, paint_primary, paint_success, paint_warning};

pub fn highlight_code(code: &str) -> String {
    let keywords = [
        "fn", "pub", "struct", "enum", "impl", "trait", "class", "def", "import", "from",
        "let", "mut", "return", "if", "else", "match", "for", "while", "in", "use",
    ];

    let mut result = String::new();
    for line in code.lines() {
        let mut highlighted_line = line.to_string();
        
        if line.trim().starts_with("//") || line.trim().starts_with("#") {
            result.push_str(&paint_muted(line));
            result.push('\n');
            continue;
        }

        for word in &keywords {
            let pattern_bound = format!(" {} ", word);
            if highlighted_line.contains(&pattern_bound) {
                highlighted_line = highlighted_line.replace(
                    &pattern_bound,
                    &format!(" {} ", word.bright_magenta().bold())
                );
            }
            
            let pattern_start = format!("{} ", word);
            if highlighted_line.starts_with(&pattern_start) {
                highlighted_line = highlighted_line.replacen(
                    &pattern_start,
                    &format!("{} ", word.bright_magenta().bold()),
                    1
                );
            }
        }

        result.push_str(&highlighted_line);
        result.push('\n');
    }
    
    if result.ends_with('\n') {
        result.pop();
    }
    result
}

pub fn print_table(headers: &[&str], rows: &[Vec<String>]) {
    if headers.is_empty() || rows.is_empty() {
        return;
    }

    let mut col_widths = vec![0; headers.len()];
    for (i, header) in headers.iter().enumerate() {
        col_widths[i] = header.len();
    }

    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            if i < col_widths.len() {
                col_widths[i] = col_widths[i].max(cell.len());
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
        .map(|(i, h)| {
            let padded = format!("{:<width$}", h, width = col_widths[i]);
            format!(" {} ", padded.bold().cyan())
        })
        .collect::<Vec<String>>()
        .join("│");
    println!("│{}│", header_line);

    println!("├{}┤", separator);

    for row in rows {
        let row_line = row
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let padded = format!("{:<width$}", c, width = col_widths[i]);
                format!(" {} ", padded)
            })
            .collect::<Vec<String>>()
            .join("│");
        println!("│{}│", row_line);
    }

    println!("└{}┘", separator.replace('┼', "┴"));
}

pub fn print_section(title: &str) {
    println!("\n{}", paint_primary(title));
}

pub fn print_success_msg(msg: &str) {
    println!("{} {}", "✓".green().bold(), paint_success(msg));
}

pub fn print_warning_msg(msg: &str) {
    println!("{} {}", "⚠".yellow().bold(), paint_warning(msg));
}
