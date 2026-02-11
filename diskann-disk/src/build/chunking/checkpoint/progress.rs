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
    fn test_progress_completed() {
        let progress = Progress::Completed;
        match progress {
            Progress::Completed => assert!(true),
            _ => panic!("Expected Completed variant"),
        }
    }

    #[test]
    fn test_progress_processed() {
        let progress = Progress::Processed(42);
        match progress {
            Progress::Processed(n) => assert_eq!(n, 42),
            _ => panic!("Expected Processed variant"),
        }
    }

    #[test]
    fn test_progress_clone() {
        let progress = Progress::Processed(100);
        let cloned = progress.clone();
        match (progress, cloned) {
            (Progress::Processed(n1), Progress::Processed(n2)) => assert_eq!(n1, n2),
            _ => panic!("Both should be Processed"),
        }
    }

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

    #[test]
    fn test_progress_debug() {
        let progress = Progress::Processed(5);
        let debug_str = format!("{:?}", progress);
        assert!(debug_str.contains("Processed"));
        assert!(debug_str.contains("5"));
    }
}
