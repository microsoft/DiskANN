/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

mod backend;
mod datafiles;
mod inputs;

use diskann_benchmark_runner as runner;

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let mut output = runner::output::default();
    cli.run(&mut output)
}

#[derive(Debug, clap::Parser)]
struct Cli {
    #[arg(long, action)]
    quiet: bool,

    #[command(flatten)]
    app: runner::App,
}

impl Cli {
    fn parse() -> Self {
        <Self as clap::Parser>::parse()
    }

    fn run(&self, output: &mut dyn runner::Output) -> anyhow::Result<()> {
        self.check_target(output)?;
        let mut registry = runner::Registry::new();
        backend::register_benchmarks(&mut registry)?;
        self.app.run(&registry, output)
    }

    #[cfg(target_arch = "x86_64")]
    fn check_target(&self, mut output: &mut dyn runner::Output) -> anyhow::Result<()> {
        use diskann_wide::Architecture;
        use std::io::Write;
        if !self.quiet
            && diskann_wide::arch::Current::level() < diskann_wide::arch::x86_64::V3::level()
        {
            writeln!(
                output,
                "WARNING: compiled without AVX2. Set RUSTFLAGS=\"-Ctarget-cpu=x86-64-v3\" for best performance. Pass --quiet to suppress."
            )?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn check_target(&self, _output: &mut dyn runner::Output) -> anyhow::Result<()> {
        Ok(())
    }
}
