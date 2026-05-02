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

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::{debug, error, info, warn};

    #[test]
    fn test_init_test_subscriber() {
        let _guard = init_test_subscriber();
        // Test that logging works without panicking
        info!("test info message");
        warn!("test warn message");
        error!("test error message");
        debug!("test debug message");
    }

    #[test]
    fn test_init_test_subscriber_guard_scope() {
        {
            let _guard = init_test_subscriber();
            info!("inside guard scope");
        }
        // After guard is dropped, we can create a new one
        let _guard2 = init_test_subscriber();
        info!("new guard scope");
    }
}
