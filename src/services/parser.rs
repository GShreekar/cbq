use std::fs;
use std::path::Path;
use tree_sitter::{Parser, Node};
use crate::services::chunker::{CodeChunk, slice_by_lines};

fn extension_to_language_name(ext: &str) -> Option<&'static str> {
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
        "cs" => Some("c_sharp"),
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

fn slice_large_chunk(chunk: CodeChunk) -> Vec<CodeChunk> {
    let lines: Vec<&str> = chunk.content.lines().collect();
    let mut sub_chunks = Vec::new();
    let chunk_size = 30;
    let overlap = 5;

    let mut start = 0;
    while start < lines.len() {
        let end = std::cmp::min(start + chunk_size, lines.len());
        let chunk_lines = &lines[start..end];
        let chunk_content = chunk_lines.join("\n");

        let sub_start_line = chunk.start_line + start;
        let sub_end_line = chunk.start_line + end - 1;

        sub_chunks.push(CodeChunk {
            file_path: chunk.file_path.clone(),
            name: format!("{}-part-{}-{}", chunk.name, sub_start_line, sub_end_line),
            chunk_type: chunk.chunk_type.clone(),
            content: chunk_content,
            start_line: sub_start_line,
            end_line: sub_end_line,
        });

        if end == lines.len() {
            break;
        }
        start += chunk_size - overlap;
    }
    sub_chunks
}

pub fn parse_file(file_path: &Path) -> Result<Vec<CodeChunk>, anyhow::Error> {
    let content = fs::read_to_string(file_path)?;
    let extension = file_path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    let lang_name = extension_to_language_name(&extension);

    if let Some(name) = lang_name {
        if let Ok(language) = tree_sitter_language_pack::get_language(name) {
            let mut parser = Parser::new();
            parser.set_language(&language)?;

            if let Some(tree) = parser.parse(&content, None) {
                let root_node = tree.root_node();
                let mut chunks = Vec::new();

                traverse_ast(root_node, &content, file_path, &mut chunks);

                if !chunks.is_empty() {
                    let mut final_chunks = Vec::new();
                    for chunk in chunks {
                        if chunk.content.lines().count() > 40 {
                            final_chunks.extend(slice_large_chunk(chunk));
                        } else {
                            final_chunks.push(chunk);
                        }
                    }
                    return Ok(final_chunks);
                }
            }
        }
    }

    Ok(slice_by_lines(file_path.to_path_buf(), &content))
}

fn traverse_ast(node: Node, source: &str, file_path: &Path, chunks: &mut Vec<CodeChunk>) {
    let node_type = node.kind();
    let is_structural = is_structural_node(node_type);

    if is_structural {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        
        if start_byte < source.len() && end_byte <= source.len() {
            let chunk_content = source[start_byte..end_byte].to_string();
            let start_line = node.start_position().row + 1;
            let end_line = node.end_position().row + 1;
            let name = extract_node_name(node, source);

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

    if !is_structural || is_container_node(node_type) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            traverse_ast(child, source, file_path, chunks);
        }
    }
}