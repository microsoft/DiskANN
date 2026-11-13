/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! A simplified subset of label-based filtering.
//!
//! * Limitations: Only supports "and", "or", and "eq" style clauses with string labels.
//! * Advantages: Natively uses an inverted index, parses JSONL directly into a semi-validated
//!   AST, avoids a dependency on `diskann-tools` and `diskann-label-filter`.

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    io::BufRead,
};

use anyhow::{Context, ensure};
use diskann_utils::views::rowmajor::{self, MatrixMut};
use serde::{
    Deserialize, Deserializer,
    de::{Error as _, IgnoredAny, MapAccess, Visitor},
};

#[derive(Debug, PartialEq, Eq, Hash)]
struct Label {
    field: String,
    value: String,
}

/// An inverted index for filtered searches.
#[derive(Debug)]
pub(crate) struct InvertedIndex {
    /// Document IDs for each literal field-value pair.
    postings: HashMap<Label, HashSet<usize>>,
    num_documents: usize,
}

impl InvertedIndex {
    /// Treat `reader` as a reader over JSONL where each entry looks like;
    /// ```text
    /// {
    ///     "doc_id": <integer>,
    ///     rest...
    /// }
    /// ```
    /// where
    ///
    /// * "doc_id" is mandatory.
    /// * `rest...` consists of an un-bounded number of key-value pairs where all values
    ///   must be strings.
    ///
    /// Within the parsed list, we require "doc_id"s to be unique and contain each value
    /// between 0 and the number of lines in `reader`.
    pub(crate) fn parse(reader: impl BufRead) -> anyhow::Result<Self> {
        let mut postings = HashMap::<Label, HashSet<usize>>::new();
        let mut ids = HashSet::new();
        on_lines(reader, |s| {
            let Document { doc_id, labels } = serde_json::from_str(s)?;
            ensure!(ids.insert(doc_id), "duplicate doc ID: {doc_id}");

            for (field, value) in labels {
                postings
                    .entry(Label { field, value })
                    .or_default()
                    .insert(doc_id);
            }
            Ok(())
        })
        .context("metadata")?;

        let num_documents = ids.len();
        ensure!(
            ids.iter().all(|id| *id < num_documents),
            "metadata doc_id values must be dense in 0..{num_documents}"
        );
        Ok(Self {
            postings,
            num_documents,
        })
    }

    /// Treat `reader` as a reader over JSONL where each entry looks like;
    /// ```text
    /// {
    ///     "query_id": <integer>,
    ///     "filter": {
    ///         <expr>
    ///     }
    /// }
    /// ```
    /// where
    ///
    /// * "query_id" is mandatory; IDs must be unique and dense in `0..number_of_queries`.
    /// * "filter" is mandatory.
    /// * `<expr>` one of:
    ///   - `{ "$and": [<expr>...] }`: An "and" clause of nested expressions.
    ///   - `{ "$or": [<expr>...] }`: An "or" clause of nested expressions.
    ///   - `{ <attribute>: { "$eq": <value> } }`: Predicate satisfied for document IDs
    ///     where `<attribute>` is equal to `<value>` (which must a string).
    ///
    /// Returns a [`rowmajor::Owned`] with one row per-query and one column per-document.
    /// Entry `(q, d)` will be `true` if and only if document `d` satisfies the predicate for
    /// query ID `q`, regardless of record order.
    pub(crate) fn queries(&self, reader: impl BufRead) -> anyhow::Result<rowmajor::Owned<bool>> {
        let mut matches = Vec::new();
        let mut ids = HashSet::new();
        on_lines(reader, |s| {
            let query: Query = serde_json::from_str(s)?;
            ensure!(
                ids.insert(query.query_id),
                "duplicate query ID: {}",
                query.query_id
            );
            matches.push((query.query_id, self.parse_query(query)?));
            Ok(())
        })
        .context("queries")?;

        let num_queries = matches.len();
        ensure!(
            ids.iter().all(|id| *id < num_queries),
            "queries query_id values must be dense in 0..{num_queries}"
        );
        let mut masks = rowmajor::Owned::try_from_element(num_queries, self.num_documents, false)?;
        for (query_id, matches) in matches {
            let row = masks.row_mut(query_id);
            for id in matches {
                row[id] = true;
            }
        }
        Ok(masks)
    }

    /// Evaluate in postorder using separate work and completed-result stacks.
    fn parse_query(&self, query: Query) -> anyhow::Result<HashSet<usize>> {
        // The stack of work that needs to be done.
        let mut stack = vec![Work::Eval(&query.filter)];

        // Partial results.
        //
        // Every `Eq` clause pushes a new entry onto this stack.
        // Reduction ops (e.g. And/Or) pop and merge the trailing number of operands and
        // push the result back onto the stack.
        //
        // At the end, we expect there to be exactly one entry in `results`, which is the
        // final result.
        let mut results = Vec::<Cow<'_, HashSet<usize>>>::new();

        // Mark a reduction operation, registring `op` and then all sub-problems.
        fn reduction_over<'a>(stack: &mut Vec<Work<'a>>, op: Op, children: &'a [Expr]) {
            stack.push(Work::Reduce {
                op,
                operands: children.len(),
            });
            stack.extend(children.iter().rev().map(Work::Eval));
        }

        while let Some(current) = stack.pop() {
            match current {
                Work::Eval(expr) => match expr {
                    Expr::And(children) => reduction_over(&mut stack, Op::And, children),
                    Expr::Or(children) => reduction_over(&mut stack, Op::Or, children),
                    Expr::Eq(label) => match self.postings.get(label) {
                        Some(matches) => results.push(Cow::Borrowed(matches)),
                        None => results.push(Cow::Owned(HashSet::new())),
                    },
                },
                Work::Reduce { op, operands } => {
                    // Merge the last `operands` operations using `op`.
                    let start = results
                        .len()
                        .checked_sub(operands)
                        .context("filter operation is missing operand results")?;

                    let mut children = results.drain(start..);
                    let mut accumulator = children
                        .next()
                        .context("filter operation has no operands")?;

                    for child in children {
                        op.merge(&mut accumulator, child);
                    }

                    results.push(accumulator);
                }
            }
        }
        ensure!(
            results.len() == 1,
            "filter evaluation must produce exactly one result"
        );
        results
            .pop()
            .context("filter evaluation has no result")
            .map(Cow::into_owned)
    }
}

//------------------//
// Document Parsing //
//------------------//

#[derive(Debug, Deserialize)]
struct Document {
    doc_id: usize,
    #[serde(flatten)]
    labels: HashMap<String, String>,
}

//---------------//
// Query Parsing //
//---------------//

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Query {
    query_id: usize,
    filter: Expr,
}

// The "grammar" supported by our filter framework.
#[derive(Debug)]
enum Expr {
    And(Vec<Expr>),
    Or(Vec<Expr>),
    Eq(Label),
}

#[derive(Deserialize)]
enum EqClause {
    #[serde(rename = "$eq")]
    Eq(String),
}

impl<'de> Deserialize<'de> for Expr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ExprVisitor;

        impl<'de> Visitor<'de> for ExprVisitor {
            type Value = Expr;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a filter object with exactly one entry")
            }

            // We expect exactly one entry consisting of:
            //
            // * "$and": `Vec<Expr>`
            // * "$or": `Vec<Expr>`
            // * <attribute>: `EqClause`.
            //
            // Everything else is rejected, as are entries with more than one key-value pair.
            // Additionally, `<attribute>` is rejected if it begins with "$".
            fn visit_map<M>(self, mut map: M) -> Result<Expr, M::Error>
            where
                M: MapAccess<'de>,
            {
                let key = map
                    .next_key::<String>()?
                    .ok_or_else(|| M::Error::custom("expected a nonempty filter object"))?;

                let expr = match key.as_str() {
                    "$and" | "$or" => {
                        let children = map.next_value::<Vec<Expr>>()?;
                        if children.is_empty() {
                            return Err(M::Error::custom(format!("{key} must not be empty")));
                        }
                        if key == "$and" {
                            Expr::And(children)
                        } else {
                            Expr::Or(children)
                        }
                    }
                    _ if key.starts_with('$') => {
                        return Err(M::Error::custom(format!(
                            "unsupported filter operator {key}"
                        )));
                    }
                    _ => {
                        let EqClause::Eq(value) = map.next_value()?;
                        Expr::Eq(Label { field: key, value })
                    }
                };

                if map.next_key::<IgnoredAny>()?.is_some() {
                    return Err(M::Error::custom(
                        "expected a filter object with exactly one entry",
                    ));
                }
                Ok(expr)
            }
        }

        deserializer.deserialize_map(ExprVisitor)
    }
}

//------------------//
// Query Evaluation //
//------------------//

#[derive(Debug)]
enum Work<'a> {
    Eval(&'a Expr),
    Reduce { op: Op, operands: usize },
}

#[derive(Debug)]
enum Op {
    And,
    Or,
}

impl Op {
    fn merge(&self, set: &mut Cow<'_, HashSet<usize>>, with: Cow<'_, HashSet<usize>>) {
        let set = set.to_mut();
        match self {
            Self::And => set.retain(|id| with.contains(id)),
            Self::Or => set.extend(with.iter().copied()),
        }
    }
}

//-------//
// Utils //
//-------//

/// Invoke the closure `f` on each line from `reader`.
fn on_lines<R, F>(reader: R, mut f: F) -> anyhow::Result<()>
where
    R: BufRead,
    F: FnMut(&str) -> anyhow::Result<()>,
{
    reader.lines().enumerate().try_for_each(|(line, s)| {
        let context = || format!("on line {}", line + 1);
        let s = s.with_context(context)?;
        f(&s).with_context(context)
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use diskann_utils::{assert_contains, views::rowmajor::Matrix};
    use serde_json::{Value, json};

    use super::*;

    /// In human readable form:
    ///
    /// | Doc ID | camera | year |
    /// +--------+--------+------+
    /// |      0 |  Canon | 2009 |
    /// |      1 |        | 2009 |
    /// |      2 |  Canon | 2010 |
    const METADATA: &str = concat!(
        "{\"doc_id\":2,\"camera\":\"Canon\",\"year\":\"2010\"}\r\n",
        "{\"doc_id\":0,\"camera\":\"Canon\",\"year\":\"2009\"}\r\n",
        "{\"doc_id\":1,\"year\":\"2009\"}"
    );

    fn generate_masks(
        metadata: impl BufRead,
        queries: impl BufRead,
    ) -> anyhow::Result<rowmajor::Owned<bool>> {
        InvertedIndex::parse(metadata)?.queries(queries)
    }

    fn masks(expressions: &[Value]) -> rowmajor::Owned<bool> {
        let queries = expressions
            .iter()
            .enumerate()
            .rev()
            .map(|(query_id, filter)| json!({"query_id": query_id, "filter": filter}).to_string())
            .collect::<Vec<_>>()
            .join("\n");

        generate_masks(METADATA.as_bytes(), queries.as_bytes()).unwrap()
    }

    #[test]
    fn evaluates_masks_by_query_id() {
        // Simplified expressions
        //
        // 0. camera == Canon
        // 1. camera == Canon && year == 2009
        // 2. country == US
        // 3. camera == NIKON
        // 4. camera == Canon && camera == NIKON
        let masks = masks(&[
            json!({"camera": {"$eq": "Canon"}}),
            json!({"$and": [
                {"camera": {"$eq": "Canon"}},
                {"$and": [{"year": {"$eq": "2009"}}]}
            ]}),
            json!({"country": {"$eq": "US"}}),
            json!({"camera": {"$eq": "NIKON"}}),
            json!({"$and": [
                {"camera": {"$eq": "Canon"}},
                {"camera": {"$eq": "NIKON"}}
            ]}),
        ]);
        assert_eq!((masks.nrows(), masks.ncols()), (5, 3));

        assert_eq!(masks.row(0), &[true, false, true]);
        assert_eq!(masks.row(1), &[true, false, false]);
        for row in 2..masks.nrows() {
            assert_eq!(masks.row(row), &[false; 3]);
        }
    }

    #[test]
    fn restores_parent_operations() {
        // Simplified expressions
        //
        // 0. camera == Canon && (year == 2009 || year == 2011)
        // 1. year == 2010 || (camera == Canon && year == 2009)
        // 2. (camera == NIKON || year == 2009) && (camera == Canon || year == 2010)
        // 3. (camera == Canon && camera == NIKON) || year == 2010
        // 4. country == US && camera == Canon
        let masks = masks(&[
            json!({"$and": [
                {"camera": {"$eq": "Canon"}},
                {"$or": [
                    {"year": {"$eq": "2009"}},
                    {"year": {"$eq": "2011"}}
                ]}
            ]}),
            json!({"$or": [
                {"year": {"$eq": "2010"}},
                {"$and": [
                    {"camera": {"$eq": "Canon"}},
                    {"year": {"$eq": "2009"}}
                ]}
            ]}),
            json!({"$and": [
                {"$or": [
                    {"camera": {"$eq": "NIKON"}},
                    {"year": {"$eq": "2009"}}
                ]},
                {"$or": [
                    {"camera": {"$eq": "Canon"}},
                    {"year": {"$eq": "2010"}}
                ]}
            ]}),
            json!({"$or": [
                {"$and": [
                    {"camera": {"$eq": "Canon"}},
                    {"camera": {"$eq": "NIKON"}}
                ]},
                {"year": {"$eq": "2010"}}
            ]}),
            json!({"$and": [
                {"country": {"$eq": "US"}},
                {"$or": [{"camera": {"$eq": "Canon"}}]}
            ]}),
        ]);

        assert_eq!(masks.row(0), &[true, false, false]);
        assert_eq!(masks.row(1), &[true, false, true]);
        assert_eq!(masks.row(2), &[true, false, false]);
        assert_eq!(masks.row(3), &[false, false, true]);
        assert_eq!(masks.row(4), &[false, false, false]);
    }

    #[test]
    fn rejects_invalid_metadata() {
        for (metadata, expected) in [
            ("{\"doc_id\":0}\n{\"doc_id\":0}", "duplicate doc ID: 0"),
            ("{\"doc_id\":1}", "must be dense"),
            ("{\"doc_id\":0}\n\n", "on line 2"), // empty line
            ("{\"doc_id\":0}\nnot json", "on line 2"), // Invalid JSON
            ("{\"doc_id\":-1}", "on line 1"),    // negative doc id
            ("{\"doc_id\":0.5}", "on line 1"),   // fractional doc id
            ("{\"doc_id\":18446744073709551616}", "on line 1"), // exceeds usize::MAX
            ("{\"doc_id\":0,\"year\":2009}", "on line 1"), // value not a string
        ] {
            let error = generate_masks(metadata.as_bytes(), &b""[..]).unwrap_err();
            let message = format!("{error:#}");

            assert_contains!(message, "metadata");
            assert_contains!(message, expected);
        }
    }

    #[test]
    fn rejects_invalid_queries() {
        for expression in [
            json!(null),
            json!({}),
            json!({"camera": {"$eq": "Canon"}, "year": {"$eq": "2009"}}),
            json!({"camera": "Canon"}),
            json!({"camera": "$eq"}),
            json!({"camera": ["Canon"]}),
            json!({"camera": {"$ne": "Canon"}}),
            json!({"camera": {"$eq": 1}}),
            json!({"camera": {"$eq": "Canon", "$ne": "NIKON"}}),
            json!({"$not": {"camera": {"$eq": "Canon"}}}),
            json!({"$and": []}),
            json!({"$and": {}}),
            json!({"$or": []}),
            json!({"$or": {}}),
            json!({"$and": [
                {"absent": {"$eq": "no matches"}},
                {"$or": [{"camera": {"$ne": "Canon"}}]}
            ]}),
            json!({"$or": [
                {"camera": {"$eq": "Canon"}},
                {"year": {"$eq": "2009"}},
                {"camera": {"$ne": "Canon"}}
            ]}),
        ] {
            let query = json!({"query_id": 0, "filter": expression}).to_string();
            let error = generate_masks(METADATA.as_bytes(), query.as_bytes()).unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("queries"), "{message}");
            assert!(message.contains("on line 1"), "{message}");
        }
        for expression in [
            r#"{"camera":{"$eq":"Canon"},"camera":{"$eq":"NIKON"}}"#,
            r#"{"$and":[{"year":{"$eq":"2009"}}],"$and":[{"year":{"$eq":"2010"}}]}"#,
            r#"{"camera":{"$eq":"Canon","$eq":"NIKON"}}"#,
        ] {
            let query = format!(r#"{{"query_id":0,"filter":{expression}}}"#);
            let error = generate_masks(METADATA.as_bytes(), query.as_bytes()).unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("queries"), "{message}");
            assert!(message.contains("on line 1"), "{message}");
        }
        let valid = "{\"query_id\":0,\"filter\":{\"year\":{\"$eq\":\"2009\"}}}";
        for invalid in ["", "not json", "{\"filter\":{}}"] {
            let queries = format!("{valid}\n{invalid}\n");
            let error = generate_masks(METADATA.as_bytes(), queries.as_bytes()).unwrap_err();
            assert!(format!("{error:#}").contains("on line 2"), "{error:#}");
        }
    }

    #[test]
    fn rejects_duplicate_or_nondense_query_ids() {
        for (ids, expected) in [
            (vec![0, 0], "duplicate query ID: 0"),
            (vec![1], "must be dense"),
            (vec![0, 2], "must be dense"),
            (vec![0, usize::MAX], "must be dense"),
        ] {
            let queries = ids
                .iter()
                .map(|query_id| {
                    json!({
                        "query_id": query_id,
                        "filter": {"camera": {"$eq": "Canon"}}
                    })
                    .to_string()
                })
                .collect::<Vec<_>>()
                .join("\n");
            let error = generate_masks(METADATA.as_bytes(), queries.as_bytes()).unwrap_err();
            let message = format!("{error:#}");
            assert!(message.contains("queries"), "{message}");
            assert!(message.contains(expected), "{message}");
            if ids == [0, 0] {
                assert!(message.contains("on line 2"), "{message}");
            }
        }
    }

    #[test]
    fn handles_empty_inputs() {
        let masks = generate_masks(METADATA.as_bytes(), &b""[..]).unwrap();
        assert_eq!((masks.nrows(), masks.ncols()), (0, 3));
        let queries = br#"{"query_id":0,"filter":{"camera":{"$eq":"Canon"}}}"#;
        let masks = generate_masks(&b""[..], &queries[..]).unwrap();
        assert_eq!((masks.nrows(), masks.ncols()), (1, 0));
        let masks = generate_masks(&b""[..], &b""[..]).unwrap();
        assert_eq!((masks.nrows(), masks.ncols()), (0, 0));
    }

    // This spot-checks that YFCC loads correctly and checks a few hard-coded queries.
    #[test]
    fn loads_yfcc_masks() {
        let root = diskann_utils::test_data_root().join("yfcc");
        let metadata = BufReader::new(File::open(root.join("yfcc_metadata.json")).unwrap());
        let queries = BufReader::new(File::open(root.join("yfcc_query_filters.json")).unwrap());
        let masks = generate_masks(metadata, queries).unwrap();

        assert_eq!((masks.nrows(), masks.ncols()), (100, 10_000));

        let metadata = BufReader::new(File::open(root.join("yfcc_metadata.json")).unwrap());
        on_lines(metadata, |s| {
            let document: Document = serde_json::from_str(s)?;
            for (row, conditions) in [
                (0, vec![("camera", "EPSON")]),
                (33, vec![("camera", "NIKON"), ("year", "2006")]),
                (66, vec![("year", "2009")]),
            ] {
                let expected = conditions.iter().all(|(field, value)| {
                    document.labels.get(*field).map(String::as_str) == Some(*value)
                });
                assert_eq!(
                    masks.row(row)[document.doc_id],
                    expected,
                    "query row {row}, doc_id {}",
                    document.doc_id
                );
            }
            Ok(())
        })
        .unwrap();
    }
}
