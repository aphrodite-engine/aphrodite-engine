// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use anyhow::Context;
use clap::Parser;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(
    name = "aphrodite-bench",
    about = "Benchmark online serving throughput",
    version
)]
struct Cli {
    #[command(flatten)]
    args: aphrodite_bench::BenchServeArgs,
}

// TODO: unify the tracing subscriber used by different binaries.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

fn main() -> anyhow::Result<()> {
    init_tracing();

    let cli = Cli::parse();

    aphrodite_bench::prepare_process();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("Failed to build tokio runtime")?;

    runtime.block_on(aphrodite_bench::run(cli.args))
}
