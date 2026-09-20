use std::path::Path;
use chrono::Local;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use crate::config::settings::cbq_home;
use crate::services::vector_search::SearchResult;

/// One question asked of a project, with the answer given and the chunks it drew on.
pub struct RecordedTurn {
    pub session_id: String,
    pub asked_at: String,
    pub question: String,
    // None when the model failed to answer.
    pub answer: Option<String>,
    pub citations: Vec<Citation>,
}

/// A chunk that an answer was based on.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Citation {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub score: f64,
}

/// Returns where a project's transcript is written when no path is given.
pub fn default_export_path(project_root: &Path) -> Result<std::path::PathBuf, anyhow::Error> {
    let project_name = project_root.file_name().map(|name| name.to_string_lossy().into_owned());
    let file_name = format!(
        "{}-{}.md",
        project_name.unwrap_or_else(|| "project".to_string()),
        Local::now().format("%Y-%m-%d_%H-%M-%S")
    );
    Ok(cbq_home()?.join("exports").join(file_name))
}

/// Returns an identifier that groups the turns of one chat session or one-off search.
pub fn new_session_id() -> String {
    Local::now().format("%Y%m%d-%H%M%S%.3f").to_string()
}

/// Returns the current time in the format history is stored and displayed in.
pub fn now_timestamp() -> String {
    Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Describes the chunks an answer cited.
pub fn citations_from(results: &[SearchResult]) -> Vec<Citation> {
    results
        .iter()
        .map(|result| Citation {
            file_path: result.chunk.file_path.to_string_lossy().into_owned(),
            start_line: result.chunk.start_line,
            end_line: result.chunk.end_line,
            score: result.score,
        })
        .collect()
}

/// Records a question, its answer and its citations in the project's own index.
pub fn record_turn(conn: &Connection, turn: &RecordedTurn) -> Result<(), anyhow::Error> {
    conn.execute(
        "INSERT INTO turns (session_id, asked_at, question, answer, citations) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            turn.session_id,
            turn.asked_at,
            turn.question,
            turn.answer,
            serde_json::to_string(&turn.citations)?
        ],
    )?;
    Ok(())
}

/// Reads the most recent turns for this project, newest first.
pub fn read_turns(conn: &Connection, limit: usize) -> Result<Vec<RecordedTurn>, anyhow::Error> {
    // Ordered by id, so turns recorded within the same second still come back in order.
    let mut stmt = conn.prepare(
        "SELECT session_id, asked_at, question, answer, citations FROM turns ORDER BY id DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |row| {
        let citations: String = row.get(4)?;
        Ok(RecordedTurn {
            session_id: row.get(0)?,
            asked_at: row.get(1)?,
            question: row.get(2)?,
            answer: row.get(3)?,
            citations: serde_json::from_str(&citations).unwrap_or_default(),
        })
    })?;
    Ok(rows.collect::<Result<_, _>>()?)
}

/// Counts every turn recorded for this project.
pub fn count_turns(conn: &Connection) -> Result<usize, anyhow::Error> {
    Ok(conn.query_row("SELECT COUNT(*) FROM turns", [], |row| row.get(0))?)
}

/// Renders turns as a Markdown transcript, oldest first, grouped into the sessions they were asked in.
pub fn render_transcript(turns: &[RecordedTurn], project_root: &Path) -> String {
    let mut transcript = format!("# cbq history for {}\n\n", project_root.display());
    transcript.push_str(&format!("Exported on {}\n", now_timestamp()));

    let mut current_session = String::new();
    for turn in turns.iter().rev() {
        if turn.session_id != current_session {
            current_session = turn.session_id.clone();
            transcript.push_str(&format!("\n---\n\n## Session {}\n", turn.session_id));
        }
        transcript.push_str(&format!("\n### {}\n*{}*\n\n", turn.question, turn.asked_at));
        match &turn.answer {
            Some(answer) => transcript.push_str(&format!("{}\n", answer)),
            None => transcript.push_str("_No answer was recorded for this question._\n"),
        }
        transcript.push_str(&render_citations(&turn.citations));
    }
    transcript
}

fn render_citations(citations: &[Citation]) -> String {
    if citations.is_empty() {
        return "\n**Sources:** none\n".to_string();
    }
    let mut rendered = "\n**Sources:**\n".to_string();
    for citation in citations {
        rendered.push_str(&format!(
            "- `{}:{}-{}` (score {:.2})\n",
            citation.file_path, citation.start_line, citation.end_line, citation.score
        ));
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::schema::create_tables;

    fn turn(session_id: &str, question: &str, answer: Option<&str>) -> RecordedTurn {
        RecordedTurn {
            session_id: session_id.to_string(),
            asked_at: "2026-09-20 10:00:00".to_string(),
            question: question.to_string(),
            answer: answer.map(str::to_string),
            citations: vec![Citation {
                file_path: "src/cart.rs".to_string(),
                start_line: 3,
                end_line: 9,
                score: 0.72,
            }],
        }
    }

    fn index_with_turns(turns: &[RecordedTurn]) -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        create_tables(&conn).unwrap();
        for turn in turns {
            record_turn(&conn, turn).unwrap();
        }
        conn
    }

    #[test]
    fn recorded_answer_is_read_back() {
        let conn = index_with_turns(&[turn("s1", "how does checkout work?", Some("It calls compute_total."))]);
        let read = read_turns(&conn, 10).unwrap();
        assert_eq!(read[0].answer.as_deref(), Some("It calls compute_total."));
    }

    #[test]
    fn citations_survive_the_round_trip() {
        let conn = index_with_turns(&[turn("s1", "q", Some("a"))]);
        let read = read_turns(&conn, 10).unwrap();
        assert_eq!(read[0].citations[0].file_path, "src/cart.rs");
        assert_eq!(read[0].citations[0].start_line, 3);
    }

    #[test]
    fn turns_are_read_newest_first() {
        let conn = index_with_turns(&[turn("s1", "first", Some("a")), turn("s1", "second", Some("b"))]);
        let questions: Vec<String> = read_turns(&conn, 10).unwrap().into_iter().map(|t| t.question).collect();
        assert_eq!(questions, vec!["second".to_string(), "first".to_string()]);
    }

    #[test]
    fn reading_is_limited_to_the_requested_count() {
        let conn = index_with_turns(&[turn("s1", "first", Some("a")), turn("s1", "second", Some("b"))]);
        assert_eq!(read_turns(&conn, 1).unwrap().len(), 1);
    }

    #[test]
    fn unanswered_turn_is_recorded_without_an_answer() {
        let conn = index_with_turns(&[turn("s1", "q", None)]);
        assert!(read_turns(&conn, 10).unwrap()[0].answer.is_none());
    }

    #[test]
    fn transcript_contains_questions_answers_and_sources() {
        let transcript = render_transcript(&[turn("s1", "how does checkout work?", Some("It calls compute_total."))], Path::new("/p"));
        assert!(transcript.contains("### how does checkout work?"));
        assert!(transcript.contains("It calls compute_total."));
        assert!(transcript.contains("`src/cart.rs:3-9` (score 0.72)"));
    }

    #[test]
    fn transcript_runs_oldest_first_and_splits_sessions() {
        let turns = vec![turn("s2", "newest", Some("b")), turn("s1", "oldest", Some("a"))];
        let transcript = render_transcript(&turns, Path::new("/p"));
        assert!(transcript.find("oldest").unwrap() < transcript.find("newest").unwrap());
        assert_eq!(transcript.matches("## Session").count(), 2);
    }
}
