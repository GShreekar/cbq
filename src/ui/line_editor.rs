use std::io::{IsTerminal, Write};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType};
use crossterm::{cursor, execute};

/// What the person typed, or how they ended the session.
pub enum Input {
    Line(String),
    /// Ctrl-C: abandon this line, keep the session.
    Interrupted,
    /// Ctrl-D on an empty line, or the end of piped input.
    EndOfInput,
}

/// Reads a line with editing and history, falling back to plain reads when input isn't a terminal.
pub struct LineEditor {
    history: Vec<String>,
    interactive: bool,
}

// The text being typed and where the cursor sits in it, kept apart from the terminal handling.
#[derive(Default)]
struct Buffer {
    characters: Vec<char>,
    cursor: usize,
}

impl LineEditor {
    pub fn new() -> Self {
        Self { history: Vec::new(), interactive: std::io::stdin().is_terminal() }
    }

    /// Reads one line, remembering it so the arrow keys can bring it back.
    pub fn read_line(&mut self, prompt: &str) -> Result<Input, anyhow::Error> {
        let input = match self.interactive {
            true => self.read_edited_line(prompt)?,
            false => read_plain_line()?,
        };
        match &input {
            Input::Line(line) if !line.trim().is_empty() && self.history.last() != Some(line) => {
                self.history.push(line.clone())
            }
            _ => {}
        }
        Ok(input)
    }

    fn read_edited_line(&mut self, prompt: &str) -> Result<Input, anyhow::Error> {
        let mut buffer = Buffer::default();
        // Walks backwards through history; past the end means "the line being typed".
        let mut recalled: Option<usize> = None;

        enable_raw_mode()?;
        let outcome = self.edit(prompt, &mut buffer, &mut recalled);
        disable_raw_mode()?;
        println!();
        outcome
    }

    fn edit(&self, prompt: &str, buffer: &mut Buffer, recalled: &mut Option<usize>) -> Result<Input, anyhow::Error> {
        buffer.redraw(prompt)?;
        loop {
            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != event::KeyEventKind::Press {
                continue;
            }

            match self.apply(key, buffer, recalled) {
                Some(input) => return Ok(input),
                None => buffer.redraw(prompt)?,
            }
        }
    }

    // Returns Some once the line is finished, None while it is still being edited.
    fn apply(&self, key: KeyEvent, buffer: &mut Buffer, recalled: &mut Option<usize>) -> Option<Input> {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match (key.code, control) {
            (KeyCode::Enter, _) => return Some(Input::Line(buffer.take())),
            (KeyCode::Char('c'), true) => return Some(Input::Interrupted),
            (KeyCode::Char('d'), true) if buffer.characters.is_empty() => return Some(Input::EndOfInput),
            (KeyCode::Char('u'), true) => buffer.clear(),
            (KeyCode::Char('a'), true) | (KeyCode::Home, _) => buffer.cursor = 0,
            (KeyCode::Char('e'), true) | (KeyCode::End, _) => buffer.cursor = buffer.characters.len(),
            (KeyCode::Char('w'), true) => buffer.delete_word(),
            (KeyCode::Char(character), false) => buffer.insert(character),
            (KeyCode::Backspace, _) => buffer.backspace(),
            (KeyCode::Delete, _) => buffer.delete(),
            (KeyCode::Left, _) => buffer.cursor = buffer.cursor.saturating_sub(1),
            (KeyCode::Right, _) => buffer.cursor = (buffer.cursor + 1).min(buffer.characters.len()),
            (KeyCode::Up, _) => self.recall(buffer, recalled, true),
            (KeyCode::Down, _) => self.recall(buffer, recalled, false),
            _ => {}
        }
        None
    }

    fn recall(&self, buffer: &mut Buffer, recalled: &mut Option<usize>, older: bool) {
        if self.history.is_empty() {
            return;
        }
        let position = match (*recalled, older) {
            (None, true) => Some(self.history.len() - 1),
            (Some(0), true) => Some(0),
            (Some(position), true) => Some(position - 1),
            (Some(position), false) if position + 1 < self.history.len() => Some(position + 1),
            (Some(_), false) => None,
            (None, false) => None,
        };
        *recalled = position;
        buffer.set(position.map(|position| self.history[position].as_str()).unwrap_or(""));
    }
}

impl Default for LineEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl Buffer {
    fn insert(&mut self, character: char) {
        self.characters.insert(self.cursor, character);
        self.cursor += 1;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.characters.remove(self.cursor - 1);
            self.cursor -= 1;
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.characters.len() {
            self.characters.remove(self.cursor);
        }
    }

    fn delete_word(&mut self) {
        while self.cursor > 0 && self.characters[self.cursor - 1].is_whitespace() {
            self.backspace();
        }
        while self.cursor > 0 && !self.characters[self.cursor - 1].is_whitespace() {
            self.backspace();
        }
    }

    fn clear(&mut self) {
        self.characters.clear();
        self.cursor = 0;
    }

    fn set(&mut self, line: &str) {
        self.characters = line.chars().collect();
        self.cursor = self.characters.len();
    }

    fn take(&mut self) -> String {
        let line: String = self.characters.iter().collect();
        self.clear();
        line
    }

    fn redraw(&self, prompt: &str) -> Result<(), anyhow::Error> {
        let line: String = self.characters.iter().collect();
        let mut stdout = std::io::stdout();
        execute!(stdout, cursor::MoveToColumn(0), Clear(ClearType::CurrentLine))?;
        write!(stdout, "{}{}", prompt, line)?;
        execute!(stdout, cursor::MoveToColumn((prompt.chars().count() + self.cursor) as u16))?;
        stdout.flush()?;
        Ok(())
    }
}

fn read_plain_line() -> Result<Input, anyhow::Error> {
    let mut line = String::new();
    match std::io::stdin().read_line(&mut line)? {
        0 => Ok(Input::EndOfInput),
        _ => Ok(Input::Line(line.trim_end_matches(['\n', '\r']).to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer_with(line: &str) -> Buffer {
        let mut buffer = Buffer::default();
        buffer.set(line);
        buffer
    }

    #[test]
    fn typing_adds_characters_at_the_cursor() {
        let mut buffer = buffer_with("helo");
        buffer.cursor = 3;
        buffer.insert('l');
        assert_eq!(buffer.take(), "hello");
    }

    #[test]
    fn backspace_removes_the_character_before_the_cursor() {
        let mut buffer = buffer_with("cart");
        buffer.backspace();
        assert_eq!(buffer.take(), "car");
    }

    #[test]
    fn delete_removes_the_character_under_the_cursor() {
        let mut buffer = buffer_with("cart");
        buffer.cursor = 0;
        buffer.delete();
        assert_eq!(buffer.take(), "art");
    }

    #[test]
    fn backspace_at_the_start_does_nothing() {
        let mut buffer = buffer_with("cart");
        buffer.cursor = 0;
        buffer.backspace();
        assert_eq!(buffer.take(), "cart");
    }

    #[test]
    fn deleting_a_word_stops_at_the_previous_space() {
        let mut buffer = buffer_with("how does checkout work");
        buffer.delete_word();
        assert_eq!(buffer.take(), "how does checkout ");
    }

    #[test]
    fn a_multibyte_line_is_edited_by_character_not_byte() {
        let mut buffer = buffer_with("café");
        buffer.backspace();
        assert_eq!(buffer.take(), "caf");
    }

    #[test]
    fn taking_the_line_leaves_the_buffer_empty() {
        let mut buffer = buffer_with("question");
        assert_eq!(buffer.take(), "question");
        assert_eq!(buffer.take(), "");
    }

    #[test]
    fn the_arrow_keys_walk_back_through_history() {
        let editor = LineEditor { history: vec!["first".to_string(), "second".to_string()], interactive: true };
        let mut buffer = Buffer::default();
        let mut recalled = None;

        editor.recall(&mut buffer, &mut recalled, true);
        assert_eq!(buffer.characters.iter().collect::<String>(), "second");
        editor.recall(&mut buffer, &mut recalled, true);
        assert_eq!(buffer.characters.iter().collect::<String>(), "first");
        editor.recall(&mut buffer, &mut recalled, false);
        assert_eq!(buffer.characters.iter().collect::<String>(), "second");
    }

    #[test]
    fn walking_forward_past_the_newest_entry_clears_the_line() {
        let editor = LineEditor { history: vec!["only".to_string()], interactive: true };
        let mut buffer = Buffer::default();
        let mut recalled = None;

        editor.recall(&mut buffer, &mut recalled, true);
        editor.recall(&mut buffer, &mut recalled, false);
        assert_eq!(buffer.characters.iter().collect::<String>(), "");
    }

    #[test]
    fn history_without_entries_leaves_the_line_alone() {
        let editor = LineEditor { history: Vec::new(), interactive: true };
        let mut buffer = buffer_with("typed");
        editor.recall(&mut buffer, &mut None, true);
        assert_eq!(buffer.take(), "typed");
    }
}
