use crate::services::vector_search::SearchResult;

// Enough of each candidate for the model to judge it, without spending the context on one chunk.
const MAX_CANDIDATE_CHARS: usize = 400;

/// Asks the model to order the candidates by how well they answer the question.
pub fn build_rerank_prompt(query: &str, candidates: &[SearchResult]) -> String {
    let listing: String = candidates
        .iter()
        .enumerate()
        .map(|(position, result)| {
            format!(
                "[{}] {} ({}:{}-{})\n{}\n\n",
                position + 1,
                result.chunk.qualified_name(),
                result.chunk.file_path.display(),
                result.chunk.start_line,
                result.chunk.end_line,
                shorten(&result.chunk.content)
            )
        })
        .collect();

    format!(
        "Order these code snippets by how well they answer the question. \
        Reply with only their numbers, best first, separated by commas, and nothing else.\n\n\
        QUESTION: {}\n\n\
        SNIPPETS:\n{}\
        ORDER:",
        query, listing
    )
}

/// Reads the model's reply as an ordering of the candidates, ignoring anything that isn't one.
pub fn parse_ranking(answer: &str, candidate_count: usize) -> Vec<usize> {
    let mut ranking = Vec::new();
    let mut digits = String::new();

    for character in answer.chars().chain(std::iter::once(' ')) {
        if character.is_ascii_digit() {
            digits.push(character);
            continue;
        }
        if let Ok(number) = digits.parse::<usize>() {
            let position = number.wrapping_sub(1);
            if number >= 1 && number <= candidate_count && !ranking.contains(&position) {
                ranking.push(position);
            }
        }
        digits.clear();
    }
    ranking
}

/// Reorders results as the model asked, keeping any it didn't mention in their original order.
pub fn apply_ranking(results: Vec<SearchResult>, ranking: &[usize], limit: usize) -> Vec<SearchResult> {
    let mut taken = vec![false; results.len()];
    let mut ordered: Vec<SearchResult> = Vec::new();

    for position in ranking {
        if let Some(result) = results.get(*position) {
            ordered.push(result.clone());
            taken[*position] = true;
        }
    }
    for (position, result) in results.into_iter().enumerate() {
        if !taken[position] {
            ordered.push(result);
        }
    }
    ordered.truncate(limit);
    ordered
}

fn shorten(content: &str) -> String {
    match content.char_indices().nth(MAX_CANDIDATE_CHARS) {
        Some((cut, _)) => format!("{}…", &content[..cut]),
        None => content.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use crate::services::chunker::CodeChunk;

    fn result(name: &str) -> SearchResult {
        SearchResult {
            chunk: CodeChunk {
                file_path: PathBuf::from(format!("src/{}.rs", name)),
                language: "rust".to_string(),
                name: name.to_string(),
                chunk_type: "function".to_string(),
                parent: None,
                content: format!("pub fn {}() {{}}", name),
                start_line: 1,
                end_line: 3,
            },
            score: 0.5,
            matched_keywords: false,
            found_via_calls: false,
        }
    }

    fn names(results: &[SearchResult]) -> Vec<String> {
        results.iter().map(|result| result.chunk.name.clone()).collect()
    }

    #[test]
    fn a_plain_list_of_numbers_is_read_as_an_order() {
        assert_eq!(parse_ranking("3, 1, 2", 3), vec![2, 0, 1]);
    }

    #[test]
    fn prose_around_the_numbers_is_ignored() {
        assert_eq!(parse_ranking("The best is [2], then [1].", 3), vec![1, 0]);
    }

    #[test]
    fn numbers_outside_the_candidate_list_are_dropped() {
        assert_eq!(parse_ranking("9, 2", 3), vec![1]);
    }

    #[test]
    fn a_repeated_number_is_only_counted_once() {
        assert_eq!(parse_ranking("2, 2, 1", 3), vec![1, 0]);
    }

    #[test]
    fn an_answer_with_no_numbers_gives_no_order() {
        assert!(parse_ranking("I cannot decide.", 3).is_empty());
    }

    #[test]
    fn results_follow_the_order_the_model_gave() {
        let results = vec![result("first"), result("second"), result("third")];
        let ordered = apply_ranking(results, &[2, 0], 3);
        assert_eq!(names(&ordered), vec!["third", "first", "second"]);
    }

    #[test]
    fn candidates_the_model_ignored_are_kept_behind_the_rest() {
        let results = vec![result("first"), result("second")];
        assert_eq!(names(&apply_ranking(results, &[], 2)), vec!["first", "second"]);
    }

    #[test]
    fn only_the_requested_number_of_results_comes_back() {
        let results = vec![result("first"), result("second"), result("third")];
        assert_eq!(apply_ranking(results, &[1], 2).len(), 2);
    }

    #[test]
    fn the_prompt_numbers_every_candidate() {
        let prompt = build_rerank_prompt("how is tax applied?", &[result("first"), result("second")]);
        assert!(prompt.contains("[1] first (src/first.rs:1-3)"));
        assert!(prompt.contains("[2] second (src/second.rs:1-3)"));
        assert!(prompt.contains("QUESTION: how is tax applied?"));
    }
}
