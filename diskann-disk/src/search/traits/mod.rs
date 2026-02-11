/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub mod vertex_provider;
pub use vertex_provider::VertexProvider;

pub mod vertex_provider_factory;
pub use vertex_provider_factory::VertexProviderFactory;

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_structure() {
        // Module structure is verified at compile time
    }
}
