use std::collections::HashMap;

/// A file and line an answer pointed at.
#[derive(Debug, PartialEq, Clone)]
pub struct Citation {
    pub path: String,
    pub line: usize,
}

/// Finds the `file:line` references an answer makes, so they can be checked against the index.
pub fn find_citations(answer: &str) -> Vec<Citation> {
    let mut citations = Vec::new();
    for word in answer.split(|character: char| character.is_whitespace() || "()[]{}<>,\"'`".contains(character)) {
        let Some(citation) = parse_citation(word) else {
            continue;
        };
        if !citations.contains(&citation) {
            citations.push(citation);
        }
    }
    citations
}

/// Names the citations that point at a file or line the index doesn't have.
pub fn unverified<'a>(citations: &'a [Citation], file_lengths: &HashMap<String, usize>) -> Vec<&'a Citation> {
    citations
        .iter()
        .filter(|citation| match file_lengths.get(&citation.path) {
            // The index records where chunks end, which is the last line cbq ever saw in that file.
            Some(last_line) => citation.line > *last_line,
            None => true,
        })
        .collect()
}

fn parse_citation(word: &str) -> Option<Citation> {
    let trimmed = word.trim_end_matches(['.', ':', ';']);
    let (path, position) = trimmed.rsplit_once(':')?;
    if path.is_empty() || !path.contains('.') || path.ends_with(':') {
        return None;
    }

    // Both "src/cart.rs:9" and "src/cart.rs:9-14" name line 9 as their start.
    let first_line = position.split(['-', '–']).next()?;
    let line: usize = first_line.parse().ok()?;
    Some(Citation { path: path.trim_start_matches("./").to_string(), line })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lengths(entries: &[(&str, usize)]) -> HashMap<String, usize> {
        entries.iter().map(|(path, last)| (path.to_string(), *last)).collect()
    }

    #[test]
    fn a_file_and_line_is_a_citation() {
        assert_eq!(find_citations("See src/cart.rs:9 for details."), vec![Citation { path: "src/cart.rs".into(), line: 9 }]);
    }

    #[test]
    fn a_line_range_is_read_from_its_start() {
        assert_eq!(find_citations("`src/cart.rs:9-14`"), vec![Citation { path: "src/cart.rs".into(), line: 9 }]);
    }

    #[test]
    fn a_leading_dot_slash_is_ignored() {
        assert_eq!(find_citations("./src/cart.rs:3"), vec![Citation { path: "src/cart.rs".into(), line: 3 }]);
    }

    #[test]
    fn the_same_citation_is_only_reported_once() {
        assert_eq!(find_citations("src/cart.rs:9 and again src/cart.rs:9").len(), 1);
    }

    #[test]
    fn prose_without_a_file_is_not_a_citation() {
        assert!(find_citations("The ratio was 3:4 and the time 10:30.").is_empty());
    }

    #[test]
    fn a_citation_inside_the_file_is_verified() {
        let citations = find_citations("src/cart.rs:9");
        assert!(unverified(&citations, &lengths(&[("src/cart.rs", 40)])).is_empty());
    }

    #[test]
    fn a_line_past_the_end_of_the_file_is_flagged() {
        let citations = find_citations("src/cart.rs:900");
        assert_eq!(unverified(&citations, &lengths(&[("src/cart.rs", 40)])).len(), 1);
    }

    #[test]
    fn a_file_the_index_has_never_seen_is_flagged() {
        let citations = find_citations("src/invented.rs:3");
        assert_eq!(unverified(&citations, &lengths(&[("src/cart.rs", 40)])).len(), 1);
    }
}
