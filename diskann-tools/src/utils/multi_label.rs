/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::collections::HashSet;

#[derive(Debug, Default)]
pub struct MultiLabel {
    base_clause: HashSet<String>,   // For base labels
    query_clause: Vec<Vec<String>>, // For query labels, first level is AND, second level is OR
}

impl MultiLabel {
    // Constructor
    pub fn new() -> Self {
        MultiLabel {
            base_clause: HashSet::new(),
            query_clause: Vec::new(),
        }
    }

    // Static method to create MultiLabel from base label
    pub fn from_base(base_label: &str) -> Self {
        let mut ml = MultiLabel::new();
        for token in base_label.split(',') {
            let token = token.trim().to_string();
            ml.base_clause.insert(token);
        }
        ml
    }

    // Static method to create MultiLabel from query label
    pub fn from_query(query_label: &str) -> Self {
        let mut ml = MultiLabel::new();
        for token in query_label.split('&') {
            let mut or_clause = Vec::new();
            for inner_token in token.split('|') {
                let inner_token = inner_token.trim().to_string();
                or_clause.push(inner_token);
            }
            ml.query_clause.push(or_clause);
        }
        ml
    }

    // Method to print the query clause
    pub fn print_query(&self) {
        for (i, and_clause) in self.query_clause.iter().enumerate() {
            for (j, or_clause) in and_clause.iter().enumerate() {
                print!("{}", or_clause);
                if j < and_clause.len() - 1 {
                    print!("|");
                }
            }
            if i < self.query_clause.len() - 1 {
                print!("&");
            }
        }
        println!();
    }

    // Method to check if the query label is a subset of the base label
    pub fn is_subset_of(&self, base_label: &MultiLabel) -> bool {
        for and_clause in &self.query_clause {
            let mut or_pass = false;
            for or_clause in and_clause {
                if base_label.base_clause.contains(or_clause) {
                    or_pass = true;
                    break;
                }
            }
            if !or_pass {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::MultiLabel;
    const BASE_LABEL:&str = "BRAND=Caltric,CAT=Automotive,CAT=MotorcyclePowersports,CAT=Parts,CAT=Filters,CAT=OilFilters,RATING=5";
    #[test]
    fn test_subset1() {
        let mut query_label = "CAT=ExteriorAccessories&RATING=4|RATING=5";

        let base_ml = MultiLabel::from_base(BASE_LABEL);
        let mut query_ml = MultiLabel::from_query(query_label);

        query_ml.print_query();

        assert!(!query_ml.is_subset_of(&base_ml));

        query_label = "CAT=Automotive&RATING=4|RATING=5";

        query_ml = MultiLabel::from_query(query_label);

        query_ml.print_query();

        assert!(query_ml.is_subset_of(&base_ml));

        query_label = "CAT=ExteriorAccessories&RATING=4|RATING=5";

        query_ml = MultiLabel::from_query(query_label);

        query_ml.print_query();

        assert!(!query_ml.is_subset_of(&base_ml));
    }
}
