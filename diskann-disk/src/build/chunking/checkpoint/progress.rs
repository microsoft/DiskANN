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
