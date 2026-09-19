use std::fs;
use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::services::chunker::{CodeChunk, MAX_CHUNK_BYTES, fit_to_byte_budget, slice_by_lines};

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

fn extract_node_name(node: Node, source: &str) -> String {
    if let Some(name_node) = node.child_by_field_name("name") {
        let n_start = name_node.start_byte();
        let n_end = name_node.end_byte();
        if n_start < source.len() && n_end <= source.len() {
            return source[n_start..n_end].to_string();
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let child_kind = child.kind();
        if child_kind == "identifier" || child_kind == "type_identifier" || child_kind == "field_identifier" {
            let n_start = child.start_byte();
            let n_end = child.end_byte();
            if n_start < source.len() && n_end <= source.len() {
                return source[n_start..n_end].to_string();
            }
        }
    }

    "anonymous".to_string()
}

pub fn parse_file(file_path: &Path) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let content = fs::read_to_string(file_path)?;
    parse_source(file_path, &content)
}

/// Splits already-read source into chunks, labelling each with `file_path`.
pub fn parse_source(file_path: &Path, content: &str) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let mut chunks = parse_syntax_chunks(file_path, content)?;
    if chunks.is_empty() {
        chunks = slice_by_lines(file_path.to_path_buf(), content);
    }

    let source_lines: Vec<&str> = content.lines().collect();
    Ok(chunks
        .into_iter()
        .flat_map(|chunk| fit_to_byte_budget(chunk, &source_lines))
        .collect())
}

fn parse_syntax_chunks(file_path: &Path, content: &str) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let extension = file_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let Some(language_name) = extension_to_language_name(&extension) else {
        return Ok(Vec::new());
    };
    // Grammars are fetched on first use; without one the file is still indexed by line windows.
    let Ok(language) = tree_sitter_language_pack::get_language(language_name) else {
        return Ok(Vec::new());
    };

    let mut parser = Parser::new();
    parser.set_language(&language)?;
    let Some(tree) = parser.parse(content, None) else {
        return Ok(Vec::new());
    };

    let mut chunks = Vec::new();
    traverse_ast(tree.root_node(), content, file_path, &mut chunks, None);
    Ok(chunks)
}

fn traverse_ast(node: Node, source: &str, file_path: &Path, chunks: &mut Vec<CodeChunk>, parent_sig: Option<&str>) {
    let node_type = node.kind();
    let is_structural = is_structural_node(node_type);
    let is_container = is_container_node(node_type);

    let mut current_sig = parent_sig.map(|s| s.to_string());
    let mut container_index = None;

    if is_structural {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        
        if start_byte < source.len() && end_byte <= source.len() {
            let mut chunk_content = source[start_byte..end_byte].to_string();
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;
            let name = extract_node_name(node, source);

            if is_container {
                container_index = Some(chunks.len());
                let text = &source[start_byte..end_byte];
                if let Some(idx) = text.find('{') {
                    current_sig = Some(source[start_byte..start_byte + idx + 1].trim().to_string());
                } else {
                    current_sig = Some(format!("{} {{", name));
                }
            }

            if let Some(sig) = parent_sig {
                if !is_container {
                    chunk_content = format!("{}\n    // ...\n{}\n}}", sig, chunk_content);
                }
            }

            chunks.push(CodeChunk {
                file_path: file_path.to_path_buf(),
                name,
                chunk_type: node_type.replace("_item", "").replace("_declaration", "").replace("_definition", ""),
                content: chunk_content,
                start_line,
                end_line,
            });
        }
    }

    if !is_structural || is_container {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            traverse_ast(child, source, file_path, chunks, current_sig.as_deref());
        }
    }

    if let Some(index) = container_index {
        let has_member_chunks = chunks.len() > index + 1;
        if has_member_chunks && chunks[index].content.len() > MAX_CHUNK_BYTES {
            chunks.remove(index); // its members are already indexed individually
        }
    }
}
