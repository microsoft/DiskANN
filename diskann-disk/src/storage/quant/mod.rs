/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod generator;
pub use generator::{GeneratorContext, QuantDataGenerator};

mod pq;
pub use pq::pq_generation::{PQGeneration, PQGenerationContext};

mod compressor;
pub use compressor::{CompressionStage, QuantCompressor};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify that key types are accessible
        let _ = core::any::type_name::<GeneratorContext>();
        let _ = core::any::type_name::<CompressionStage>();
    }
}
