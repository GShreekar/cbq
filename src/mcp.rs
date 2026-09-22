use std::path::{Path, PathBuf};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, BufReader};
use crate::config::settings::{load_config_for, Config};
use crate::db::index_metadata::ensure_index_model_matches;
use crate::db::location::find_indexed_project;
use crate::db::queries::{enclosing_symbol, find_definitions, find_references};
use crate::db::schema::open_index;
use crate::doctor::summarise_index;
use crate::services::embeddings::embed_query;
use crate::services::ollama::{is_same_model, Ollama};
use crate::services::search::find_relevant_chunks;
use crate::services::vector_search::{load_chunk_vectors, ChunkVectors};

// The version of the protocol cbq was written against; a client asking for another is answered in its own.
const PROTOCOL_VERSION: &str = "2024-11-05";
const DEFAULT_SEARCH_LIMIT: usize = 5;

/// Serves the index over the Model Context Protocol, so editors and agents can query it as a tool.
pub async fn serve(directory: &Path) -> Result<(), anyhow::Error> {
    let mut server = Server::open(directory).await?;
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    // One JSON-RPC message per line, in and out. Nothing else may touch stdout.
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(request) => server.respond(request).await,
            Err(err) => Some(failure(Value::Null, -32700, &format!("Invalid JSON: {}", err))),
        };
        if let Some(response) = response {
            println!("{}", response);
        }
    }
    Ok(())
}

struct Server {
    config: Config,
    project_root: PathBuf,
    connection: rusqlite::Connection,
    vectors: ChunkVectors,
    ollama: Ollama,
}

impl Server {
    async fn open(directory: &Path) -> Result<Self, anyhow::Error> {
        let config = load_config_for(directory)?;
        let project = find_indexed_project(directory)?
            .ok_or_else(|| anyhow::anyhow!("No index covers {}; run `cbq index` first", directory.display()))?;
        let connection = open_index(&project.db_path)?;
        ensure_index_model_matches(&connection, &config.ollama.embedding_model)?;

        let ollama = Ollama::new(&config.ollama.host, config.ollama.port)?;
        // Models are checked, never pulled: a pull would write progress over the protocol stream.
        let installed = ollama.installed_models().await?;
        anyhow::ensure!(
            installed.iter().any(|name| is_same_model(name, &config.ollama.embedding_model)),
            "Embedding model '{}' is not on {}; pull it before starting the server",
            config.ollama.embedding_model,
            ollama.address()
        );

        let vectors = load_chunk_vectors(&connection)?;
        Ok(Self { config, project_root: project.root, connection, vectors, ollama })
    }

    // Returns None for notifications, which a JSON-RPC server must not answer.
    async fn respond(&mut self, request: Value) -> Option<Value> {
        let id = request.get("id").cloned();
        let method = request.get("method").and_then(Value::as_str).unwrap_or_default().to_string();
        let id = id?;

        let response = match method.as_str() {
            "initialize" => success(id.clone(), self.describe_server(&request)),
            "ping" => success(id.clone(), json!({})),
            "tools/list" => success(id.clone(), json!({ "tools": tool_definitions() })),
            "tools/call" => self.call_tool(id.clone(), request.get("params")).await,
            _ => failure(id.clone(), -32601, &format!("Unknown method '{}'", method)),
        };
        Some(response)
    }

    fn describe_server(&self, request: &Value) -> Value {
        let requested = request
            .get("params")
            .and_then(|params| params.get("protocolVersion"))
            .and_then(Value::as_str)
            .unwrap_or(PROTOCOL_VERSION);
        json!({
            "protocolVersion": requested,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "cbq", "version": env!("CARGO_PKG_VERSION") },
            "instructions": format!(
                "Searches the codebase indexed at {}. Use search_code for questions in prose, \
                 find_definition and find_references for exact symbol names.",
                self.project_root.display()
            ),
        })
    }

    async fn call_tool(&mut self, id: Value, params: Option<&Value>) -> Value {
        let Some(params) = params else {
            return failure(id, -32602, "Missing parameters");
        };
        let name = params.get("name").and_then(Value::as_str).unwrap_or_default();
        let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

        match self.run_tool(name, &arguments).await {
            Ok(result) => success(id, tool_output(&result, false)),
            Err(err) => success(id, tool_output(&json!({ "error": err.to_string() }), true)),
        }
    }

    async fn run_tool(&mut self, name: &str, arguments: &Value) -> Result<Value, anyhow::Error> {
        match name {
            "search_code" => self.search(arguments).await,
            "find_definition" => self.definition(arguments),
            "find_references" => self.references(arguments),
            "project_status" => self.status(),
            _ => anyhow::bail!("Unknown tool '{}'", name),
        }
    }

    async fn search(&mut self, arguments: &Value) -> Result<Value, anyhow::Error> {
        let query = required_text(arguments, "query")?;
        let limit = arguments.get("limit").and_then(Value::as_u64).unwrap_or(DEFAULT_SEARCH_LIMIT as u64) as usize;

        let query_vector = embed_query(&self.ollama, &self.config.ollama.embedding_model, query).await?;
        let results = find_relevant_chunks(&self.connection, &self.vectors, query, &query_vector, limit)?;
        Ok(json!({
            "query": query,
            "results": results.iter().map(crate::output::search_result_json).collect::<Vec<_>>(),
        }))
    }

    fn definition(&self, arguments: &Value) -> Result<Value, anyhow::Error> {
        let symbol = required_text(arguments, "symbol")?;
        let definitions = find_definitions(&self.connection, symbol)?;
        Ok(json!({
            "symbol": symbol,
            "definitions": definitions.iter().map(|definition| json!({
                "path": definition.file_path.to_string_lossy(),
                "start_line": definition.start_line,
                "end_line": definition.end_line,
                "kind": definition.chunk_type,
                "parent": definition.parent,
                "code": definition.content,
            })).collect::<Vec<_>>(),
        }))
    }

    fn references(&self, arguments: &Value) -> Result<Value, anyhow::Error> {
        let symbol = required_text(arguments, "symbol")?;
        let only_calls = arguments.get("only_calls").and_then(Value::as_bool).unwrap_or(false);

        let mut described = Vec::new();
        for reference in find_references(&self.connection, symbol, only_calls)? {
            let path = reference.file_path.to_string_lossy().into_owned();
            described.push(json!({
                "path": path,
                "line": reference.line,
                "kind": reference.kind,
                "caller": enclosing_symbol(&self.connection, &path, reference.line)?,
            }));
        }
        Ok(json!({ "symbol": symbol, "references": described }))
    }

    fn status(&self) -> Result<Value, anyhow::Error> {
        let summary = summarise_index(&self.project_root)?
            .ok_or_else(|| anyhow::anyhow!("The index has gone away; run `cbq index`"))?;
        Ok(json!({
            "project": summary.project_root,
            "files": summary.indexed_files,
            "chunks": summary.total_chunks,
            "languages": summary.languages,
            "stale": summary.is_stale(),
        }))
    }
}

/// The tools cbq offers: retrieval, not generation, since whatever is calling is already a model.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "search_code",
            "description": "Search the indexed codebase by meaning and by exact wording. Use for questions \
                            in prose, such as 'where is tax applied?'. Returns the matching code. Results \
                            marked found_via_calls are supporting definitions the matches call, not matches \
                            themselves, and come after them.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "What to look for, in prose or as code" },
                    "limit": { "type": "integer", "description": "How many matches to rank (default 5); a \
                                                                  couple of called definitions may follow them" }
                },
                "required": ["query"]
            }
        }),
        json!({
            "name": "find_definition",
            "description": "Show where a symbol is defined, with its code. Accepts a qualified name such \
                            as Cart::add. Instant: reads the index, not a model.",
            "inputSchema": {
                "type": "object",
                "properties": { "symbol": { "type": "string", "description": "The symbol name" } },
                "required": ["symbol"]
            }
        }),
        json!({
            "name": "find_references",
            "description": "Show everywhere a symbol is used, with the symbol that uses it. Set only_calls \
                            to true for calls without imports.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": { "type": "string", "description": "The symbol name" },
                    "only_calls": { "type": "boolean", "description": "Leave out imports" }
                },
                "required": ["symbol"]
            }
        }),
        json!({
            "name": "project_status",
            "description": "Describe the indexed project: how many files and chunks it holds, which \
                            languages, and whether the index has fallen behind the files on disk.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
    ]
}

fn required_text<'a>(arguments: &'a Value, name: &str) -> Result<&'a str, anyhow::Error> {
    arguments
        .get(name)
        .and_then(Value::as_str)
        .filter(|text| !text.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("'{}' is required", name))
}

fn tool_output(value: &Value, is_error: bool) -> Value {
    json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(value).unwrap_or_default() }],
        "isError": is_error,
    })
}

fn success(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn failure(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_declares_a_name_description_and_schema() {
        for tool in tool_definitions() {
            assert!(tool["name"].is_string());
            assert!(tool["description"].is_string());
            assert_eq!(tool["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn the_search_tool_requires_a_query() {
        let search = tool_definitions().into_iter().find(|tool| tool["name"] == "search_code").unwrap();
        assert_eq!(search["inputSchema"]["required"], json!(["query"]));
    }

    #[test]
    fn a_successful_response_carries_the_request_id() {
        let response = success(json!(7), json!({ "ok": true }));
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 7);
        assert_eq!(response["result"]["ok"], true);
    }

    #[test]
    fn a_failed_response_carries_a_code_and_message() {
        let response = failure(json!(2), -32601, "Unknown method 'x'");
        assert_eq!(response["error"]["code"], -32601);
        assert_eq!(response["error"]["message"], "Unknown method 'x'");
    }

    #[test]
    fn tool_output_is_json_inside_a_text_block() {
        let output = tool_output(&json!({ "results": [] }), false);
        assert_eq!(output["content"][0]["type"], "text");
        assert!(output["content"][0]["text"].as_str().unwrap().contains("results"));
        assert_eq!(output["isError"], false);
    }

    #[test]
    fn a_failing_tool_is_reported_as_an_error_result() {
        assert_eq!(tool_output(&json!({ "error": "boom" }), true)["isError"], true);
    }

    #[test]
    fn a_missing_argument_is_rejected() {
        assert!(required_text(&json!({}), "query").is_err());
        assert!(required_text(&json!({ "query": "  " }), "query").is_err());
        assert_eq!(required_text(&json!({ "query": "tax" }), "query").unwrap(), "tax");
    }
}
