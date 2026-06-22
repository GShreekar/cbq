use colored::{Color, Colorize};

pub const PRIMARY_COLOR: Color = Color::BrightCyan;
pub const SUCCESS_COLOR: Color = Color::BrightGreen;
pub const WARNING_COLOR: Color = Color::BrightYellow;
pub const ERROR_COLOR: Color = Color::BrightRed;
pub const MUTED_COLOR: Color = Color::TrueColor { r: 120, g: 120, b: 120 };

pub fn paint_primary(text: &str) -> String {
    text.color(PRIMARY_COLOR).bold().to_string()
}

pub fn paint_success(text: &str) -> String {
    text.color(SUCCESS_COLOR).bold().to_string()
}

pub fn paint_warning(text: &str) -> String {
    text.color(WARNING_COLOR).to_string()
}

pub fn paint_error(text: &str) -> String {
    text.color(ERROR_COLOR).bold().to_string()
}

pub fn paint_muted(text: &str) -> String {
    text.color(MUTED_COLOR).to_string()
}
