/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

#[derive(Debug, Clone)]
pub enum Progress {
    Completed,
    Processed(usize /* items processed */),
}

impl Progress {
    pub fn map(self, f: impl Fn(usize) -> usize) -> Progress {
        match self {
            Progress::Completed => Progress::Completed,
            Progress::Processed(progress) => Progress::Processed(f(progress)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_map_processed() {
        let progress = Progress::Processed(10);
        let mapped = progress.map(|n| n * 2);
        match mapped {
            Progress::Processed(n) => assert_eq!(n, 20),
            _ => panic!("Expected Processed variant"),
        }
    }

    #[test]
    fn test_progress_map_completed() {
        let progress = Progress::Completed;
        let mapped = progress.map(|n| n * 2);
        match mapped {
            Progress::Completed => assert!(true),
            _ => panic!("Expected Completed variant"),
        }
    }
}
