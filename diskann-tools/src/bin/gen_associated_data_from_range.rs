/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use clap::Parser;
use diskann_providers::storage::FileStorageProvider;
use diskann_tools::utils::{gen_associated_data_from_range, CMDResult};

fn main() -> CMDResult<()> {
    let storage_provider = FileStorageProvider;
    let args = GenAssociatedDataFromRangeArgs::parse();
    gen_associated_data_from_range(
        &storage_provider,
        &args.associated_data_path,
        args.start,
        args.end,
    )
}

#[derive(Debug, Parser)]
struct GenAssociatedDataFromRangeArgs {
    #[arg(long = "associated_data_path")]
    pub associated_data_path: String,

    #[arg(long = "start")]
    pub start: u32,

    #[arg(long = "end")]
    pub end: u32,
}
