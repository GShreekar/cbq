use std::io::{IsTerminal, Write};

/// Prints a Markdown answer as it streams in, so text appears while the model is still writing.
pub struct StreamingMarkdown {
    pending_line: String,
    // Fenced code renders correctly only as a whole, so its lines are held until the closing fence.
    code_block: Option<String>,
    render_markdown: bool,
}

impl Default for StreamingMarkdown {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingMarkdown {
    /// Renders Markdown when writing to a terminal, and leaves piped output as plain text.
    pub fn new() -> Self {
        Self {
            pending_line: String::new(),
            code_block: None,
            render_markdown: std::io::stdout().is_terminal(),
        }
    }

    /// Takes the next piece of streamed text and prints whatever is now complete.
    pub fn push(&mut self, text: &str) {
        self.pending_line.push_str(text);
        while let Some(newline) = self.pending_line.find('\n') {
            let line: String = self.pending_line.drain(..=newline).collect();
            self.print(line.trim_end_matches('\n'));
        }
    }

    /// Prints whatever is left once the answer ends.
    pub fn finish(&mut self) {
        let leftover = std::mem::take(&mut self.pending_line);
        if !leftover.is_empty() {
            self.print(&leftover);
        }
        // An answer that stops inside a code block still has to be shown.
        if let Some(unfinished) = self.code_block.take() {
            self.render(&unfinished);
        }
    }

    fn print(&mut self, line: &str) {
        if !self.render_markdown {
            println!("{}", line);
            return;
        }
        if let Some(block) = self.absorb_code_block(line) {
            self.render(&block);
        }
    }

    // Returns the text ready to render, or None while a code block is still open.
    fn absorb_code_block(&mut self, line: &str) -> Option<String> {
        let is_fence = line.trim_start().starts_with("```");
        match (&mut self.code_block, is_fence) {
            (Some(block), true) => {
                block.push_str(line);
                self.code_block.take()
            }
            (Some(block), false) => {
                block.push_str(line);
                block.push('\n');
                None
            }
            (None, true) => {
                self.code_block = Some(format!("{}\n", line));
                None
            }
            (None, false) => Some(line.to_string()),
        }
    }

    fn render(&self, text: &str) {
        termimad::print_text(text);
        // Terminals show the answer as it arrives only if each piece is flushed.
        let _ = std::io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain_renderer() -> StreamingMarkdown {
        StreamingMarkdown { pending_line: String::new(), code_block: None, render_markdown: true }
    }

    #[test]
    fn a_line_split_across_chunks_is_held_until_complete() {
        let mut renderer = plain_renderer();
        renderer.pending_line.push_str("half a ");
        renderer.pending_line.push_str("line");
        assert!(renderer.pending_line.find('\n').is_none());
    }

    #[test]
    fn plain_line_renders_immediately() {
        assert_eq!(plain_renderer().absorb_code_block("some text"), Some("some text".to_string()));
    }

    #[test]
    fn code_block_is_held_until_its_closing_fence() {
        let mut renderer = plain_renderer();
        assert_eq!(renderer.absorb_code_block("```rust"), None);
        assert_eq!(renderer.absorb_code_block("fn main() {}"), None);
        assert_eq!(renderer.absorb_code_block("```"), Some("```rust\nfn main() {}\n```".to_string()));
    }

    #[test]
    fn text_after_a_code_block_renders_immediately_again() {
        let mut renderer = plain_renderer();
        renderer.absorb_code_block("```");
        renderer.absorb_code_block("```");
        assert_eq!(renderer.absorb_code_block("after"), Some("after".to_string()));
    }

    #[test]
    fn unfinished_code_block_is_still_printed_at_the_end() {
        let mut renderer = plain_renderer();
        renderer.absorb_code_block("```rust");
        renderer.absorb_code_block("fn main() {}");
        assert_eq!(renderer.code_block.as_deref(), Some("```rust\nfn main() {}\n"));
    }
}
