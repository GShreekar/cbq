use std::fs;
use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::services::chunker::{CodeChunk, fit_to_byte_budget, slice_by_lines};

/// Maps a lowercase file extension to its tree-sitter grammar name.
pub fn extension_to_language_name(ext: &str) -> Option<&'static str> {
    match ext {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" => Some("javascript"),
        "jsx" => Some("javascript"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "mjs" | "cjs" => Some("javascript"),
        "go" => Some("go"),
        "c" => Some("c"),
        "cpp" | "cc" | "cxx" => Some("cpp"),
        "h" | "hpp" | "hh" => Some("cpp"),
        "java" => Some("java"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "cs" => Some("csharp"),
        "html" => Some("html"),
        "css" => Some("css"),
        "sh" | "bash" | "zsh" => Some("bash"),
        "swift" => Some("swift"),
        "kt" | "kts" => Some("kotlin"),
        "yaml" | "yml" => Some("yaml"),
        "sql" => Some("sql"),
        "toml" => Some("toml"),
        "json" => Some("json"),
        _ => None,
    }
}

fn is_container_node(kind: &str) -> bool {
    kind.contains("class") 
        || kind.contains("impl") 
        || kind.contains("trait") 
        || kind.contains("interface") 
        || kind.contains("struct")
        || kind.contains("enum")
}

fn is_structural_node(kind: &str) -> bool {
    match kind {
        "struct" | "class" | "interface" | "enum" | "impl" | "trait" | "function" | "method" | "fn" | "def" | "func" => return false,
        _ => {}
    }

    let exact_match = match kind {
        "function_item" | "struct_item" | "impl_item" | "trait_item" | "enum_item" => true,
        "function_definition" | "class_definition" | "method_definition" => true,
        "function_declaration" | "method_declaration" | "class_declaration" | "struct_declaration" | "interface_declaration" | "enum_declaration" | "type_declaration" => true,
        _ => false,
    };
    
    if exact_match {
        return true;
    }

    if kind.contains("body") 
        || kind.contains("expression") 
        || kind.contains("statement") 
        || kind.contains("call") 
        || kind.contains("argument") 
        || kind.contains("parameter")
        || kind.contains("list")
        || kind.contains("variant")
        || kind.contains("field")
        || kind.contains("type")
        || kind.contains("clause")
    {
        return false;
    }

    kind.contains("function")
        || kind.contains("method")
        || kind.contains("class")
        || kind.contains("struct")
        || kind.contains("interface")
        || kind.contains("impl")
        || kind.contains("trait")
        || kind.contains("enum")
}

// How far up the tree to look for the name of an anonymous function expression.
const MAX_BINDING_DEPTH: usize = 3;
// Caps on the synthetic module chunk, so a file with hundreds of imports stays a summary.
const MAX_MODULE_HEADER_LINES: usize = 60;
const MAX_MODULE_SYMBOLS: usize = 40;

struct ParseContext<'a> {
    source: &'a str,
    file_path: &'a Path,
    language: &'a str,
}

impl ParseContext<'_> {
    fn traverse(&self, node: Node, parent: Option<&str>, chunks: &mut Vec<CodeChunk>) {
        let is_structural = is_structural_node(node.kind());
        let is_container = is_container_node(node.kind());
        let mut child_parent = parent.map(str::to_string);

        let chunk = is_structural.then(|| self.build_chunk(node, parent, is_container)).flatten();
        if let Some(chunk) = chunk {
            if is_container {
                child_parent = Some(chunk.name.clone());
            }
            chunks.push(chunk);
        }

        // A function's insides are part of that function's chunk; a container's members are not.
        if !is_structural || is_container {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                self.traverse(child, child_parent.as_deref(), chunks);
            }
        }
    }

    fn build_chunk(&self, node: Node, parent: Option<&str>, is_container: bool) -> Option<CodeChunk> {
        let (name, anchor) = self.name_and_anchor(node);
        let first_node = leading_prefix_start(anchor, self.source);
        let start_byte = first_node.start_byte();
        let end_byte = node.end_byte().max(anchor.end_byte());
        if start_byte >= end_byte || end_byte > self.source.len() {
            return None;
        }

        let content = match is_container {
            true => container_skeleton(node, self.source, start_byte),
            false => self.source[start_byte..end_byte].to_string(),
        };
        Some(CodeChunk {
            file_path: self.file_path.to_path_buf(),
            language: self.language.to_string(),
            name,
            chunk_type: chunk_type_of(node.kind()),
            parent: parent.map(str::to_string),
            content,
            start_line: first_node.start_position().row + 1,
            end_line: node.end_position().row + 1,
        })
    }

    fn name_and_anchor<'t>(&self, node: Node<'t>) -> (String, Node<'t>) {
        if let Some(name) = declared_name(node, self.source) {
            return (name, node);
        }
        binding_name(node, self.source).unwrap_or_else(|| ("anonymous".to_string(), node))
    }
}

fn chunk_type_of(kind: &str) -> String {
    kind.replace("_item", "").replace("_declaration", "").replace("_definition", "")
}

fn node_text<'a>(node: Node, source: &'a str) -> &'a str {
    source.get(node.start_byte()..node.end_byte()).unwrap_or("")
}

fn declared_name(node: Node, source: &str) -> Option<String> {
    if let Some(name_node) = node.child_by_field_name("name") {
        return Some(node_text(name_node, source).to_string());
    }
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .find(|child| matches!(child.kind(), "identifier" | "type_identifier" | "field_identifier"))
        .map(|child| node_text(child, source).to_string())
}

// `const debounce = (fn) => {}` keeps the name on the declaration rather than on the function itself.
fn binding_name<'t>(node: Node<'t>, source: &str) -> Option<(String, Node<'t>)> {
    let mut current = node;
    for _ in 0..MAX_BINDING_DEPTH {
        let parent = current.parent()?;
        let name_node = match parent.kind() {
            "variable_declarator" | "public_field_definition" | "field_definition" => parent.child_by_field_name("name"),
            "assignment_expression" | "assignment" => parent.child_by_field_name("left"),
            "pair" => parent.child_by_field_name("key"),
            _ => None,
        };
        if let Some(name_node) = name_node {
            let name = node_text(name_node, source).trim_matches(['"', '\'']).to_string();
            return Some((name, declaration_anchor(parent)));
        }
        current = parent;
    }
    None
}

// The chunk should start at `export const debounce = ...`, not partway through the expression.
fn declaration_anchor(node: Node) -> Node {
    let mut anchor = node;
    while let Some(parent) = anchor.parent() {
        let wraps_declaration = matches!(
            parent.kind(),
            "lexical_declaration" | "variable_declaration" | "export_statement" | "expression_statement"
        );
        if !wraps_declaration {
            break;
        }
        anchor = parent;
    }
    anchor
}

// Doc comments, decorators and attributes describe the node that follows them, but sit outside it.
fn leading_prefix_start<'t>(node: Node<'t>, source: &str) -> Node<'t> {
    let mut first = node;
    while let Some(previous) = first.prev_sibling() {
        if !is_attached_prefix(previous.kind()) {
            break;
        }
        let gap = source.get(previous.end_byte()..first.start_byte()).unwrap_or(" ");
        // A blank line in between means the comment belongs to whatever came before it.
        if !gap.trim().is_empty() || gap.matches('\n').count() > 1 {
            break;
        }
        if !starts_its_own_line(source, previous.start_byte()) {
            break;
        }
        first = previous;
    }
    first
}

fn is_attached_prefix(kind: &str) -> bool {
    kind.contains("comment") || kind.contains("decorator") || kind.contains("attribute")
}

fn starts_its_own_line(source: &str, byte: usize) -> bool {
    source.get(..byte).and_then(|text| text.rsplit('\n').next()).is_none_or(|line| line.trim().is_empty())
}

// Members are indexed as chunks of their own, so a container keeps only their signatures.
fn container_skeleton(node: Node, source: &str, start_byte: usize) -> String {
    let mut skeleton = String::new();
    let mut copied_to = start_byte;
    for body in member_bodies(node) {
        if body.start_byte() < copied_to {
            continue;
        }
        skeleton.push_str(source.get(copied_to..body.start_byte()).unwrap_or(""));
        skeleton.push_str(match node_text(body, source).starts_with('{') {
            true => "{ ... }",
            false => "...",
        });
        copied_to = body.end_byte();
    }
    skeleton.push_str(source.get(copied_to..node.end_byte()).unwrap_or(""));
    skeleton
}

fn member_bodies<'t>(container: Node<'t>) -> Vec<Node<'t>> {
    let mut cursor = container.walk();
    let holder = container
        .named_children(&mut cursor)
        .find(|child| is_member_list(child.kind()))
        .unwrap_or(container);

    let mut holder_cursor = holder.walk();
    holder
        .named_children(&mut holder_cursor)
        .filter_map(|member| body_of(unwrap_definition(member)))
        .collect()
}

fn is_member_list(kind: &str) -> bool {
    kind.ends_with("_list") || kind.ends_with("_body") || kind == "block"
}

// A decorated definition wraps the definition the decorators apply to.
fn unwrap_definition(node: Node) -> Node {
    if is_structural_node(node.kind()) {
        return node;
    }
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .find(|child| is_structural_node(child.kind()))
        .unwrap_or(node)
}

fn body_of(node: Node) -> Option<Node> {
    if let Some(body) = node.child_by_field_name("body") {
        return Some(body);
    }
    let mut cursor = node.walk();
    node.named_children(&mut cursor).find(|child| is_body_kind(child.kind()))
}

fn is_body_kind(kind: &str) -> bool {
    matches!(kind, "block" | "statement_block" | "compound_statement" | "declaration_list" | "class_body")
}

// Imports, constants and the list of symbols answer "what is in this file?", which no other chunk covers.
fn module_header_chunk(root: Node, context: &ParseContext, chunks: &[CodeChunk]) -> Option<CodeChunk> {
    let mut preamble = Vec::new();
    let mut last_line = 1;
    let mut in_preamble = true;

    let mut cursor = root.walk();
    for child in root.named_children(&mut cursor) {
        let is_comment = child.kind().contains("comment");
        // Only comments at the very top describe the module; later ones describe what follows them.
        let is_module_doc = in_preamble && is_comment;
        if !is_comment {
            in_preamble = false;
        }
        if is_module_doc || is_import(child.kind()) || is_module_constant(child) {
            preamble.push(node_text(child, context.source).trim().to_string());
            last_line = last_line.max(child.end_position().row + 1);
        }
        if preamble.len() >= MAX_MODULE_HEADER_LINES {
            break;
        }
    }

    let symbols: Vec<String> = chunks
        .iter()
        .take(MAX_MODULE_SYMBOLS)
        .map(|chunk| format!("{} ({})", chunk.qualified_name(), chunk.chunk_type))
        .collect();
    if preamble.is_empty() && symbols.is_empty() {
        return None;
    }

    let mut content = format!("Module: {}\n", context.file_path.display());
    if !preamble.is_empty() {
        content.push_str(&format!("\n{}\n", preamble.join("\n")));
    }
    if !symbols.is_empty() {
        content.push_str(&format!("\nDefines: {}\n", symbols.join(", ")));
    }

    Some(CodeChunk {
        file_path: context.file_path.to_path_buf(),
        language: context.language.to_string(),
        name: context.file_path.file_name()?.to_string_lossy().into_owned(),
        chunk_type: "module".to_string(),
        parent: None,
        content,
        start_line: 1,
        end_line: last_line,
    })
}

fn is_import(kind: &str) -> bool {
    kind.contains("import") || kind.contains("use_declaration") || kind.contains("package")
        || kind.contains("using") || kind.contains("include")
}

fn is_module_constant(node: Node) -> bool {
    let kind = node.kind();
    if kind.contains("const") || kind.contains("static") || kind.contains("type_alias") || kind == "type_item" {
        return true;
    }
    // A JavaScript `const` holding a function is already indexed as that function.
    matches!(kind, "lexical_declaration" | "variable_declaration") && !holds_a_function(node)
}

fn holds_a_function(node: Node) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).any(|child| {
        let mut value_cursor = child.walk();
        child.kind().contains("function")
            || child.named_children(&mut value_cursor).any(|value| {
                value.kind().contains("function") || value.kind().contains("arrow")
            })
    })
}

pub fn parse_file(file_path: &Path) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let content = fs::read_to_string(file_path)?;
    parse_source(file_path, &content)
}

/// Splits already-read source into chunks, labelling each with `file_path`.
pub fn parse_source(file_path: &Path, content: &str) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let extension = file_path.extension().and_then(|name| name.to_str()).unwrap_or("").to_lowercase();
    let language = extension_to_language_name(&extension);

    let mut chunks = parse_syntax_chunks(file_path, content, language)?;
    if chunks.is_empty() {
        chunks = slice_by_lines(file_path.to_path_buf(), language.unwrap_or(&extension), content);
    }

    let source_lines: Vec<&str> = content.lines().collect();
    Ok(chunks
        .into_iter()
        .flat_map(|chunk| fit_to_byte_budget(chunk, &source_lines))
        .collect())
}

fn parse_syntax_chunks(
    file_path: &Path,
    content: &str,
    language: Option<&'static str>,
) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let Some(language_name) = language else {
        return Ok(Vec::new());
    };
    // Grammars are fetched on first use; without one the file is still indexed by line windows.
    let Ok(grammar) = tree_sitter_language_pack::get_language(language_name) else {
        return Ok(Vec::new());
    };

    let mut parser = Parser::new();
    parser.set_language(&grammar)?;
    let Some(tree) = parser.parse(content, None) else {
        return Ok(Vec::new());
    };

    let context = ParseContext { source: content, file_path, language: language_name };
    let mut chunks = Vec::new();
    context.traverse(tree.root_node(), None, &mut chunks);

    if let Some(header) = module_header_chunk(tree.root_node(), &context, &chunks) {
        chunks.insert(0, header);
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    // These parse with real grammars, which the language pack fetches on first use.
    fn chunks_of(file_name: &str, source: &str) -> Vec<CodeChunk> {
        parse_source(Path::new(file_name), source).unwrap()
    }

    fn named<'a>(chunks: &'a [CodeChunk], name: &str) -> &'a CodeChunk {
        chunks.iter().find(|chunk| chunk.name == name).unwrap_or_else(|| panic!("no chunk named {}", name))
    }

    const RUST_CART: &str = "\
use std::fmt;

pub const MAX_RETRIES: usize = 3;

/// Computes the total price.
pub fn compute_total(items: &[f64]) -> f64 {
    items.iter().sum()
}

impl Cart {
    /// Adds an item.
    pub fn add(&mut self, price: f64) {
        self.items.push(price);
    }
}
";

    #[test]
    fn a_functions_doc_comment_is_part_of_its_chunk() {
        let chunks = chunks_of("cart.rs", RUST_CART);
        let chunk = named(&chunks, "compute_total");
        assert!(chunk.content.starts_with("/// Computes the total price."));
    }

    #[test]
    fn a_doc_comment_is_not_taken_from_the_item_it_follows() {
        let chunks = chunks_of("cart.rs", "fn first() {}\n\n// belongs to second\nfn second() {}\n");
        assert!(!named(&chunks, "first").content.contains("belongs to second"));
        assert!(named(&chunks, "second").content.contains("belongs to second"));
    }

    #[test]
    fn a_rust_attribute_stays_with_its_item() {
        let chunks = chunks_of("cart.rs", "#[derive(Debug)]\npub struct Cart {\n    pub items: Vec<f64>,\n}\n");
        assert!(named(&chunks, "Cart").content.starts_with("#[derive(Debug)]"));
    }

    #[test]
    fn a_type_keeps_its_member_signatures_but_not_their_bodies() {
        let chunks = chunks_of("cart.rs", RUST_CART);
        let chunk = named(&chunks, "Cart");
        assert!(chunk.content.contains("pub fn add(&mut self, price: f64) { ... }"));
        assert!(!chunk.content.contains("self.items.push(price);"));
    }

    #[test]
    fn a_method_is_indexed_with_its_type_as_parent() {
        let chunks = chunks_of("cart.rs", RUST_CART);
        let chunk = named(&chunks, "add");
        assert_eq!(chunk.parent.as_deref(), Some("Cart"));
        assert!(chunk.content.contains("self.items.push(price);"));
    }

    #[test]
    fn a_struct_without_methods_keeps_its_fields() {
        let chunks = chunks_of("cart.rs", "pub struct Cart {\n    pub items: Vec<f64>,\n}\n");
        assert!(named(&chunks, "Cart").content.contains("pub items: Vec<f64>"));
    }

    #[test]
    fn an_arrow_function_takes_the_name_it_is_assigned_to() {
        let chunks = chunks_of("util.js", "// Debounce helper.\nconst debounce = (fn, ms) => { return fn; };\n");
        let chunk = named(&chunks, "debounce");
        assert!(chunk.content.starts_with("// Debounce helper.\nconst debounce ="));
    }

    #[test]
    fn an_exported_arrow_function_is_named_too() {
        let chunks = chunks_of("util.js", "export const formatPrice = (cents) => `$${cents}`;\n");
        assert!(named(&chunks, "formatPrice").content.starts_with("export const formatPrice"));
    }

    #[test]
    fn a_python_decorator_stays_with_its_function() {
        let source = "from flask import Flask\n\n@app.route(\"/users\")\ndef list_users():\n    return []\n";
        let chunks = chunks_of("users.py", source);
        let chunk = named(&chunks, "list_users");
        assert!(chunk.content.starts_with("@app.route(\"/users\")"));
    }

    #[test]
    fn a_python_class_keeps_method_signatures_only() {
        let source = "class User:\n    def name(self):\n        return self.first\n";
        let chunks = chunks_of("user.py", source);
        let chunk = named(&chunks, "User");
        assert!(chunk.content.contains("def name(self):"));
        assert!(!chunk.content.contains("return self.first"));
    }

    #[test]
    fn every_file_gets_a_module_chunk_listing_its_imports_and_symbols() {
        let chunks = chunks_of("cart.rs", RUST_CART);
        let module = named(&chunks, "cart.rs");
        assert_eq!(module.chunk_type, "module");
        assert!(module.content.contains("use std::fmt;"));
        assert!(module.content.contains("pub const MAX_RETRIES: usize = 3;"));
        assert!(module.content.contains("compute_total (function)"));
        assert!(module.content.contains("Cart::add (function)"));
    }

    #[test]
    fn a_module_chunk_does_not_steal_the_first_functions_doc_comment() {
        let chunks = chunks_of("cart.rs", RUST_CART);
        let module = named(&chunks, "cart.rs");
        assert!(!module.content.contains("Computes the total price"));
    }

    #[test]
    fn a_file_with_no_grammar_is_still_split_into_chunks() {
        let chunks = chunks_of("notes.md", "# Heading\n\nSome prose about the project.\n");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chunk_type, "general");
    }
}
