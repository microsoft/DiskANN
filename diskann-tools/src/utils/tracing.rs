/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use tracing;
use tracing_subscriber::{filter::LevelFilter, fmt, prelude::*, EnvFilter};

/// Create a default subscriber logging messages to `stdout` and respecting the `RUST_LOG`
/// environment variable.
///
/// If the environment variable is not set - then the "info" level will be used.
pub fn init_subscriber() {
    let fmt_layer = fmt::layer().with_target(true);

    let filter_layer = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}

/// Create a subscriber for the integration tests.
///
/// This subscriber returns a `Guard` that will only install the subscriber locally,
/// allowing test threads to have non-conflicting subscribers.
pub fn init_test_subscriber() -> tracing::subscriber::DefaultGuard {
    let fmt_layer = fmt::layer().with_target(true).with_test_writer();

    let filter_layer = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .set_default()
}
