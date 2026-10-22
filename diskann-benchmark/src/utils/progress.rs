/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann_benchmark_core as dbc;
use diskann_benchmark_runner as dbr;
use indicatif::{ProgressBar, ProgressStyle};

pub(crate) struct Progress<'a> {
    output: &'a mut dyn dbr::Output,
}

impl<'a> Progress<'a> {
    pub(crate) fn new(output: &'a mut dyn dbr::Output) -> Self {
        Self { output }
    }
}

impl dbc::build::AsProgress for Progress<'_> {
    fn as_progress(&self, max: usize) -> Arc<dyn dbc::build::Progress> {
        let target = self.output.draw_target();
        let meter = ProgressBar::with_draw_target(Some(max as u64), target);
        meter.set_style(
            ProgressStyle::with_template("Building [{elapsed_precise}] {wide_bar} {percent}")
                .expect("This format should be valid"),
        );
        Arc::new(Meter { meter })
    }
}

#[derive(Debug)]
struct Meter {
    meter: ProgressBar,
}

impl dbc::build::Progress for Meter {
    fn progress(&self, handled: usize) {
        self.meter.inc(handled as u64)
    }
    fn finish(&self) {
        self.meter.finish()
    }
}
