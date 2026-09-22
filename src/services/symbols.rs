use std::path::{Path, PathBuf};
use tree_sitter::Node;

/// A place where some name is used, which is how cbq knows what depends on what.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub symbol_name: String,
    pub file_path: PathBuf,
    pub line: usize,
    pub kind: &'static str,
}

pub const CALL: &str = "call";
pub const IMPORT: &str = "import";

/// Finds every call and import in a parsed file.
pub fn extract_references(root: Node, source: &str, file_path: &Path) -> Vec<Reference> {
    let mut references = Vec::new();
    collect(root, source, file_path, &mut references);
    references.dedup();
    references
}

fn collect(node: Node, source: &str, file_path: &Path, references: &mut Vec<Reference>) {
    let line = node.start_position().row + 1;
    if is_call(node.kind()) {
        if let Some(name) = called_name(node, source) {
            references.push(Reference { symbol_name: name, file_path: file_path.to_path_buf(), line, kind: CALL });
        }
    } else if is_import(node.kind()) {
        for name in imported_names(node, source) {
            references.push(Reference { symbol_name: name, file_path: file_path.to_path_buf(), line, kind: IMPORT });
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect(child, source, file_path, references);
    }
}

// Grammars name this differently: call_expression, call, method_invocation, invocation_expression.
fn is_call(kind: &str) -> bool {
    kind.contains("call") || kind.contains("invocation")
}

fn is_import(kind: &str) -> bool {
    kind.contains("import") || kind == "use_declaration" || kind.contains("require")
}

// The callee is the last name in the callee expression: `self.cart.add(x)` calls `add`.
fn called_name(node: Node, source: &str) -> Option<String> {
    let callee = node
        .child_by_field_name("function")
        .or_else(|| node.child_by_field_name("name"))
        .or_else(|| node.named_child(0))?;
    last_identifier(callee, source)
}

fn imported_names(node: Node, source: &str) -> Vec<String> {
    let mut names = Vec::new();
    collect_identifiers(node, source, &mut names);
    names
}

fn collect_identifiers(node: Node, source: &str, names: &mut Vec<String>) {
    // Composite names such as `crate::cart::compute_total` are themselves "identifier" nodes,
    // so only childless nodes count: recursing gives the final segment rather than the whole path.
    if node.named_child_count() == 0 && is_identifier(node.kind()) {
        names.push(node_text(node, source).to_string());
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_identifiers(child, source, names);
    }
}

fn last_identifier(node: Node, source: &str) -> Option<String> {
    let mut names = Vec::new();
    collect_identifiers(node, source, &mut names);
    names.pop()
}

fn is_identifier(kind: &str) -> bool {
    kind.ends_with("identifier") || kind == "constant" || kind == "word"
}

fn node_text<'a>(node: Node, source: &'a str) -> &'a str {
    source.get(node.start_byte()..node.end_byte()).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::parser::parse_source;

    // These parse with real grammars, which the language pack fetches on first use.
    fn references_in(file_name: &str, source: &str) -> Vec<Reference> {
        parse_source(Path::new(file_name), source).unwrap().references
    }

    fn names_of(references: &[Reference], kind: &str) -> Vec<String> {
        references.iter().filter(|r| r.kind == kind).map(|r| r.symbol_name.clone()).collect()
    }

    #[test]
    fn a_plain_call_is_recorded() {
        let references = references_in("cart.rs", "fn total() -> f64 {\n    compute_total(&items)\n}\n");
        assert!(names_of(&references, CALL).contains(&"compute_total".to_string()));
    }

    #[test]
    fn a_method_call_records_the_method_name() {
        let references = references_in("cart.rs", "fn add(&mut self) {\n    self.items.push(price);\n}\n");
        assert!(names_of(&references, CALL).contains(&"push".to_string()));
    }

    #[test]
    fn a_path_qualified_call_records_the_final_name() {
        let references = references_in("cart.rs", "fn run() {\n    crate::cart::compute_total(&items);\n}\n");
        assert!(names_of(&references, CALL).contains(&"compute_total".to_string()));
    }

    #[test]
    fn the_line_of_the_call_is_recorded() {
        let references = references_in("cart.rs", "fn run() {\n\n    compute_total(&items);\n}\n");
        let call = references.iter().find(|r| r.symbol_name == "compute_total").unwrap();
        assert_eq!(call.line, 3);
    }

    #[test]
    fn a_rust_import_records_what_was_imported() {
        let references = references_in("cart.rs", "use crate::cart::compute_total;\n");
        assert!(names_of(&references, IMPORT).contains(&"compute_total".to_string()));
    }

    #[test]
    fn a_javascript_named_import_is_recorded() {
        let references = references_in("app.js", "import { debounce } from './util';\n");
        assert!(names_of(&references, IMPORT).contains(&"debounce".to_string()));
    }

    #[test]
    fn a_python_import_and_call_are_both_recorded() {
        let references = references_in("app.py", "from models import User\n\ndef run():\n    return User.all()\n");
        assert!(names_of(&references, IMPORT).contains(&"User".to_string()));
        assert!(names_of(&references, CALL).contains(&"all".to_string()));
    }

    #[test]
    fn a_javascript_method_call_records_the_property_name() {
        let references = references_in("app.js", "function run() {\n  window.setTimeout(fn, 10);\n}\n");
        assert!(names_of(&references, CALL).contains(&"setTimeout".to_string()));
    }

    #[test]
    fn nested_calls_are_all_recorded() {
        let references = references_in("cart.rs", "fn run() {\n    outer(inner(x));\n}\n");
        let calls = names_of(&references, CALL);
        assert!(calls.contains(&"outer".to_string()) && calls.contains(&"inner".to_string()));
    }

    #[test]
    fn a_file_with_no_calls_has_no_references() {
        assert!(references_in("cart.rs", "pub struct Cart {\n    pub items: Vec<f64>,\n}\n").is_empty());
    }
}
