/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use anyhow::{Result, anyhow};
use azure_core::{credentials::TokenCredential, time::OffsetDateTime};
use azure_identity::AzureCliCredential;
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use loader::DatasetLoader;
use redis::{
    AsyncConnectionConfig, AsyncTypedCommands, IntoConnectionInfo, Pipeline, ToRedisArgs,
    aio::MultiplexedConnection,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio::{
    fs::File,
    io::AsyncReadExt,
    sync::mpsc,
    task::{JoinHandle, JoinSet},
};

use crate::{
    catalog::Catalog,
    garnet::Garnet,
    runbook::Runbook,
    runner::{Filter, Runner},
};

mod catalog;
mod dataset;
mod driver;
mod garnet;
mod loader;
mod report;
mod runbook;
mod runner;
#[cfg(test)]
mod test_utils;

const DEFAULT_PORT: u16 = 6379;

/// redis-rs defaults to a 500ms response timeout, which a pipelined batch of searches or
/// inserts will always exceed.
fn connection_config() -> AsyncConnectionConfig {
    AsyncConnectionConfig::new()
        .set_response_timeout(None)
        .set_connection_timeout(Some(Duration::from_secs(30)))
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElementType {
    F32,
    U8,
    I8,
    U32,
}

pub trait Element: bytemuck::Pod + Default + std::fmt::Debug + Send + Sync + 'static {
    const ELEMENT_TYPE: ElementType;
}

impl Element for f32 {
    const ELEMENT_TYPE: ElementType = ElementType::F32;
}

impl Element for u8 {
    const ELEMENT_TYPE: ElementType = ElementType::U8;
}

impl Element for i8 {
    const ELEMENT_TYPE: ElementType = ElementType::I8;
}

impl Element for u32 {
    const ELEMENT_TYPE: ElementType = ElementType::U32;
}
#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct Config {
    ips: Vec<String>,
    port: Option<u16>,
    secure: bool,
    scope: Option<String>,
    username: Option<String>,
    dataset_search_paths: Option<Vec<PathBuf>>,
}

#[derive(Parser)]
struct Options {
    /// Path to config file
    #[arg(short = 'C', long, value_name = "CONFIG_FILE")]
    config: PathBuf,

    /// Maximum number of threads for the runtime
    #[arg(short, long)]
    threads: Option<usize>,

    /// Quantizer
    #[arg(long, default_value_t)]
    quantizer: Quantizer,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ping the server
    Ping,
    /// Ingest vectors to a vector set
    Ingest(IngestArgs),
    /// Delete vector set and flush database
    Delete(DeleteArgs),
    /// Run queries and calculate recall
    Query(QueryArgs),
    /// Run runbook
    Run(RunArgs),
}

#[derive(Args)]
struct IngestArgs {
    /// Vector set key
    #[arg(long, value_name = "VECTOR_SET", default_value = "vs0")]
    set: String,

    /// Number of parallel insert tasks
    #[arg(short, long)]
    tasks: Option<usize>,

    /// Number of pipelined commands to the server
    #[arg(long, default_value = "64")]
    pipeline_size: usize,

    /// Number of start points
    #[arg(long, default_value = "10")]
    start_points: usize,

    /// Graph degree
    #[arg(long, default_value = "16")]
    degree: usize,

    /// Candidate list size during build
    #[arg(long, default_value = "100")]
    l_build: usize,

    /// Limit amount of vectors to ingest
    #[arg(long)]
    limit: Option<usize>,

    /// Input vector bin has no header
    #[arg(long)]
    no_header_with_dim: Option<usize>,

    /// Metric
    #[arg(long)]
    metric: Option<DistanceMetric>,

    /// Paths to base vectors
    base_path: PathBuf,
}

#[derive(Args)]
struct DeleteArgs {
    /// Vector set key
    #[arg(short, long, value_name = "VECTOR_SET", default_value = "vs0")]
    set: String,
}

#[derive(Args)]
struct QueryArgs {
    /// Vector set key
    #[arg(short, long, value_name = "VECTOR_SET", default_value = "vs0")]
    set: String,

    /// Number of parallel search tasks
    #[arg(short, long)]
    tasks: Option<usize>,

    /// Number of pipelined commands to the server
    #[arg(long, default_value = "64")]
    pipeline_size: usize,

    /// Candidate list size during search
    #[arg(long, default_value = "15")]
    l_search: usize,

    /// Number of ground truth neighbors to score against (the k in k-recall@n)
    #[arg(short, long, default_value = "10")]
    k: usize,

    /// Number of search results to return (the n in k-recall@n)
    #[arg(short, long, default_value = "10")]
    n: usize,

    /// Total queries to run (default: all given queries from input a single time)
    #[arg(long)]
    total_queries: Option<usize>,

    /// Path to query vectors
    query_path: PathBuf,

    /// Path to ground truth for queries
    gt_path: PathBuf,
}

#[derive(Args)]
struct RunArgs {
    /// Vector set key prefix
    #[arg(short, long, value_name = "VECTOR_SET", default_value = "vs0")]
    set: String,

    /// Number of parallel search tasks
    #[arg(
        short,
        long,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    tasks: Option<usize>,

    /// Number of pipelined commands to the server (searches are never pipelined)
    #[arg(
        long,
        default_value_t = 64,
        value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..)
    )]
    pipeline_size: usize,

    /// Graph degree
    #[arg(long, default_value = "16")]
    degree: usize,

    /// Candidate list size during build
    #[arg(long, default_value = "15")]
    l_build: usize,

    /// Candidate list size during search
    #[arg(long, default_value = "15")]
    l_search: usize,

    /// Number of ground truth neighbors to score against (the k in k-recall@n)
    #[arg(short, long, default_value = "10")]
    k: usize,

    /// Number of search results to return (the n in k-recall@n)
    #[arg(short, long, default_value = "10")]
    n: usize,

    /// Include dataset filter
    #[arg(long)]
    include: Vec<String>,

    /// Exclude dataset filter
    #[arg(long)]
    exclude: Vec<String>,

    /// Repeat search steps
    #[arg(long, default_value_t = 5, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..))]
    search_repetitions: usize,

    /// Output directory for reports
    #[arg(long, default_value = "reports")]
    report_path: PathBuf,

    /// Runbook to execute
    runbook: PathBuf,

    /// Dataset catalog directory
    catalog: PathBuf,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum DataType {
    Uint8,
    Int8,
    Float32,
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Default, Serialize)]
enum Quantizer {
    /// f32 vectors; no quantization
    #[default]
    NoQuant,
    /// f32 vectors; spherical 1-bit quantization
    Bin,
    /// f32 vectors; minmax 8-bit scalar quantization
    Q8,
    /// u8 vectors; no quantization
    XNoQuant_U8,
    /// i8 vectors; no quantization
    XNoQuant_I8,
    /// u8 vectors; spherical 1-bit quantization
    XBin_U8,
    /// i8 vectors; spherical 1-bit quantization
    XBin_I8,
}

impl Quantizer {
    pub fn data_type(&self) -> DataType {
        match self {
            Quantizer::NoQuant | Quantizer::Bin | Quantizer::Q8 => DataType::Float32,
            Quantizer::XNoQuant_I8 | Quantizer::XBin_I8 => DataType::Int8,
            Quantizer::XNoQuant_U8 | Quantizer::XBin_U8 => DataType::Uint8,
        }
    }
}

impl ToRedisArgs for Quantizer {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + redis::RedisWrite,
    {
        let q = match self {
            Quantizer::NoQuant => b"NOQUANT".as_slice(),
            Quantizer::Bin => b"BIN".as_slice(),
            Quantizer::Q8 => b"Q8".as_slice(),
            Quantizer::XNoQuant_U8 => b"XNOQUANT_U8",
            Quantizer::XNoQuant_I8 => b"XNOQUANT_I8",
            Quantizer::XBin_U8 => b"XBIN_U8",
            Quantizer::XBin_I8 => b"XBIN_I8",
        };
        out.write_arg(q);
    }
}

impl std::fmt::Display for Quantizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Quantizer::NoQuant => "NOQUANT",
            Quantizer::Bin => "BIN",
            Quantizer::Q8 => "Q8",
            Quantizer::XNoQuant_U8 => "XNOQUANT_U8",
            Quantizer::XNoQuant_I8 => "XNOQUANT_I8",
            Quantizer::XBin_U8 => "XBIN_U8",
            Quantizer::XBin_I8 => "XBIN_I8",
        };
        f.write_str(s)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Serialize, Deserialize)]
enum DistanceMetric {
    #[serde(rename = "l2", alias = "L2")]
    L2,
    #[serde(rename = "cosine", alias = "COSINE")]
    Cosine,
    #[serde(rename = "cosine_normalized", alias = "COSINE_NORMALIZED")]
    CosineNormalized,
    #[serde(
        rename = "innerproduct",
        alias = "InnerProduct",
        alias = "ip",
        alias = "INNERPRODUCT",
        alias = "inner_product",
        alias = "INNER_PRODUCT"
    )]
    InnerProduct,
}

impl ToRedisArgs for DistanceMetric {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + redis::RedisWrite,
    {
        let q = match self {
            DistanceMetric::L2 => b"L2".as_slice(),
            DistanceMetric::Cosine => b"COSINE".as_slice(),
            DistanceMetric::CosineNormalized => b"XCOSINE_NORMALIZED".as_slice(),
            DistanceMetric::InnerProduct => b"IP".as_slice(),
        };
        out.write_arg(q);
    }
}

struct VectorId(u32);

impl ToRedisArgs for VectorId {
    fn write_redis_args<W>(&self, out: &mut W)
    where
        W: ?Sized + redis::RedisWrite,
    {
        out.write_arg(bytemuck::bytes_of(&self.0));
    }
}

#[derive(Debug, Error)]
pub enum ExpiringCredentialError {
    #[error("redis error: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("azure error: {0}")]
    Azure(#[from] azure_core::Error),
}

#[derive(Clone)]
struct ExpiringCredential {
    scope: String,
    username: String,
    cred: Arc<AzureCliCredential>,
    expires: OffsetDateTime,
}

impl ExpiringCredential {
    fn new(
        scope: String,
        username: String,
        cred: Arc<AzureCliCredential>,
        expires: OffsetDateTime,
    ) -> Self {
        Self {
            scope,
            username,
            cred,
            expires,
        }
    }

    async fn refresh_if_needed(
        mut self,
        con: &mut MultiplexedConnection,
    ) -> std::result::Result<Self, ExpiringCredentialError> {
        if self.expires - OffsetDateTime::now_utc() < Duration::from_secs(300) {
            let res = self.cred.get_token(&[&self.scope], None).await?;

            redis::cmd("AUTH")
                .arg(&self.username)
                .arg(res.token.secret().to_string())
                .exec_async(con)
                .await?;

            println!(
                "DEBUG: refreshed token; new on expires in {}",
                res.expires_on
            );

            self.expires = res.expires_on;
        }

        Ok(self)
    }
}

fn main() -> Result<()> {
    let opts = Options::parse();

    let parallelism = opts
        .threads
        .unwrap_or(thread::available_parallelism()?.get());

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(parallelism)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async_main(opts))
}

async fn async_main(opts: Options) -> Result<()> {
    let mut config_file = File::open(&opts.config).await?;
    let mut contents = Vec::new();
    config_file.read_to_end(&mut contents).await?;
    let config: Config = toml::from_slice(&contents)?;

    let mut addrs = Vec::new();
    for ip in config.ips.iter().cloned() {
        let addr = if config.secure {
            redis::ConnectionAddr::TcpTls {
                host: ip,
                port: config.port.unwrap_or(DEFAULT_PORT),
                insecure: true,
                tls_params: None,
            }
        } else {
            redis::ConnectionAddr::Tcp(ip, config.port.unwrap_or(DEFAULT_PORT))
        };
        addrs.push(addr);
    }

    let (password, expires, scope) = if config.username.is_some() {
        let credentials = AzureCliCredential::new(None)?;
        let Some(scope) = &config.scope else {
            return Err(anyhow!("missing scope in config"));
        };
        let res = credentials.get_token(&[scope], None).await?;
        (
            Some(res.token.secret().to_string()),
            Some(res.expires_on),
            Some(scope.to_owned()),
        )
    } else {
        (None, None, None)
    };

    let mut infos = Vec::new();
    for addr in addrs.into_iter() {
        let mut redis_info = redis::RedisConnectionInfo::default();
        if let Some(username) = config.username.as_ref() {
            redis_info = redis_info.set_username(username.clone());
        }
        if let Some(password) = password.as_ref() {
            redis_info = redis_info.set_password(password.clone());
        }
        let info = addr.into_connection_info()?.set_redis_settings(redis_info);
        infos.push(info);
    }

    let cred = if config.username.is_some() {
        Some(ExpiringCredential::new(
            scope.unwrap(),
            config.username.clone().unwrap(),
            AzureCliCredential::new(None)?,
            expires.unwrap(),
        ))
    } else {
        None
    };

    match opts.quantizer.data_type() {
        DataType::Uint8 => dispatch::<u8>(&config, &opts.command, &opts, infos, cred).await,
        DataType::Int8 => dispatch::<i8>(&config, &opts.command, &opts, infos, cred).await,
        DataType::Float32 => dispatch::<f32>(&config, &opts.command, &opts, infos, cred).await,
    }
}

async fn dispatch<T: Element>(
    config: &Config,
    command: &Commands,
    opts: &Options,
    infos: Vec<redis::ConnectionInfo>,
    cred: Option<ExpiringCredential>,
) -> Result<()> {
    match command {
        Commands::Ping => ping(infos[0].clone()).await?,
        Commands::Ingest(args) => ingest::<T>(opts, args, infos[0].clone(), cred).await?,
        Commands::Delete(args) => delete(args, infos[0].clone()).await?,
        Commands::Query(args) => query::<T>(opts, args, infos, cred).await?,
        Commands::Run(args) => run::<T>(config, opts, args, infos, cred).await?,
    }

    Ok(())
}

async fn ping(info: redis::ConnectionInfo) -> Result<()> {
    let client = redis::Client::open(info).unwrap();
    let mut con = client
        .get_multiplexed_async_connection_with_config(&connection_config())
        .await?;

    println!("PING...");
    let result = con.ping().await?;
    println!("...{result}");

    Ok(())
}

async fn ingest<T: Element>(
    opts: &Options,
    args: &IngestArgs,
    info: redis::ConnectionInfo,
    cred: Option<ExpiringCredential>,
) -> Result<()> {
    let ds = DatasetLoader::new(&args.base_path).await?;
    let parallelism = args.tasks.unwrap_or(thread::available_parallelism()?.get());
    let vset = Arc::new(args.set.clone());
    let (tx, mut rx) = mpsc::channel(parallelism);
    let mut tasks = JoinSet::<Result<()>>::new();
    let total_vectors = ds.len();

    tasks.spawn(async move {
        let progress = ProgressBar::with_draw_target(
            Some(total_vectors as u64),
            ProgressDrawTarget::stderr_with_hz(1),
        );
        progress.set_style(ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} {elapsed}/{eta} {per_sec}",
        )?);

        while let Some(count) = rx.recv().await {
            progress.inc(count as u64);
        }

        Ok(())
    });

    let start_time = Instant::now();

    // Insert base vectors
    for _ in 0..parallelism {
        let client = redis::Client::open(info.clone())?;
        let mut con = client
            .get_multiplexed_async_connection_with_config(&connection_config())
            .await?;
        let ds = ds.clone();
        let pipeline_size = args.pipeline_size;
        let vset = vset.clone();
        let tx = tx.clone();
        let limit = args.limit.unwrap_or(ds.len());
        let l_build = args.l_build;
        let degree = args.degree;
        let mut cred = cred.clone();
        let data_type = opts.quantizer.data_type();
        let quantizer = opts.quantizer;
        let metric = args.metric;

        tasks.spawn(async move {
            let mut buf = vec![T::zeroed(); ds.batch_size() * ds.dim()];
            let mut pipeline = Pipeline::with_capacity(pipeline_size);
            let mut ingested = 0;

            loop {
                if let Some(c) = cred {
                    cred = Some(c.refresh_if_needed(&mut con).await?);
                }

                let (count, first_id) = ds.next(&mut buf).await?;
                if count == 0 {
                    return Ok(());
                }

                let mut next = 0;
                while next < count {
                    pipeline.clear();

                    let queue_size = (count - next).min(pipeline_size);
                    for i in next..next + queue_size {
                        let element = VectorId((first_id + i) as u32);
                        let buf_start = i * ds.dim();
                        let buf_end = buf_start + ds.dim();

                        pipeline.cmd("VADD").arg(&vset);

                        match data_type {
                            DataType::Uint8 => {
                                pipeline.arg(b"XU8");
                            }
                            DataType::Int8 => {
                                pipeline.arg(b"XI8");
                            }
                            DataType::Float32 => {
                                pipeline.arg(b"FP32");
                            }
                        }

                        pipeline
                            .arg(bytemuck::cast_slice::<_, u8>(&buf[buf_start..buf_end]))
                            .arg(element);

                        pipeline.arg(quantizer);

                        if let Some(metric) = metric {
                            pipeline.arg(b"XDISTANCE_METRIC").arg(metric);
                        }

                        pipeline
                            .arg(b"EF")
                            .arg(l_build.to_string().as_bytes())
                            .arg(b"M")
                            .arg(degree.to_string().as_bytes());
                    }

                    next += queue_size;

                    pipeline.exec_async(&mut con).await?;
                }

                tx.send(count).await?;

                ingested += count;
                if ingested > limit {
                    return Ok(());
                }

                if count < ds.batch_size() {
                    return Ok(());
                }
            }
        });
    }

    drop(tx);

    for result in tasks.join_all().await {
        result?;
    }

    let build_time = start_time.elapsed().as_secs_f64();

    println!("RESULTS ({total_vectors} vectors ingested in {build_time:0.2}s):");
    println!("    vps: {:0.2}", total_vectors as f64 / build_time);

    Ok(())
}

async fn delete(args: &DeleteArgs, info: redis::ConnectionInfo) -> Result<()> {
    let client = redis::Client::open(info).unwrap();
    let mut con = client
        .get_multiplexed_async_connection_with_config(&connection_config())
        .await?;

    con.del(&args.set).await?;
    Ok(())
}

async fn query<T: Element>(
    opts: &Options,
    args: &QueryArgs,
    infos: Vec<redis::ConnectionInfo>,
    cred: Option<ExpiringCredential>,
) -> Result<()> {
    let parallelism = args.tasks.unwrap_or(thread::available_parallelism()?.get());
    let vset = Arc::new(args.set.clone());
    let (tx, mut rx) = mpsc::channel(parallelism);

    let queries = DatasetLoader::<T>::load(&args.query_path).await?;
    let truth = DatasetLoader::<T>::load_groundtruth(&args.gt_path).await?;
    let total_queries = args.total_queries.unwrap_or(queries.len());

    let mut tasks = JoinSet::<Result<Vec<(usize, usize, usize, Duration)>>>::new();

    let progress_task: JoinHandle<Result<()>> = tokio::spawn(async move {
        let progress = ProgressBar::with_draw_target(
            Some(total_queries as u64),
            ProgressDrawTarget::stderr_with_hz(1),
        );
        progress.set_style(ProgressStyle::with_template(
            "{wide_bar} {pos}/{len} {elapsed}/{eta} {per_sec}",
        )?);

        while let Some(count) = rx.recv().await {
            progress.inc(count as u64);
        }

        Ok(())
    });

    let time_start = Instant::now();
    for task_idx in 0..parallelism {
        let client = redis::Client::open(infos[task_idx % infos.len()].clone())?;
        let mut con = client
            .get_multiplexed_async_connection_with_config(&connection_config())
            .await?;
        let pipeline_size = args.pipeline_size;
        let tx = tx.clone();
        let queries = queries.clone();
        let vset = vset.clone();
        let truth = truth.clone();
        let start_q = task_idx * queries.len() / parallelism;
        let total_queries = args.total_queries.unwrap_or(queries.len());
        let batch_size = total_queries.div_ceil(parallelism);
        let k = args.k;
        let n = args.n;
        let l_search = args.l_search;
        let mut cred = cred.clone();
        let data_type = opts.quantizer.data_type();

        tasks.spawn(async move {
            let mut pipeline = Pipeline::with_capacity(pipeline_size);

            let batches = batch_size.div_ceil(pipeline_size);
            let mut stats = Vec::with_capacity(batches);
            for batch_idx in 0..batches {
                if let Some(c) = cred {
                    cred = Some(c.refresh_if_needed(&mut con).await?);
                }

                pipeline.clear();

                let start_time = Instant::now();

                let vset = vset.clone();
                for pipeline_idx in 0..pipeline_size {
                    let q = (start_q + batch_idx * pipeline_size + pipeline_idx) % queries.len();
                    let qv = &*queries[q];

                    pipeline.cmd("VSIM").arg(&vset);

                    match data_type {
                        DataType::Uint8 => pipeline.arg(b"XU8"),
                        DataType::Int8 => pipeline.arg(b"XI8"),
                        DataType::Float32 => pipeline.arg(b"FP32"),
                    };

                    pipeline
                        .arg(bytemuck::cast_slice::<T, u8>(qv))
                        .arg(b"COUNT")
                        .arg(n.to_string().as_bytes())
                        .arg(b"EF")
                        .arg(l_search.to_string().as_bytes());
                }

                let results: Vec<Vec<[u8; 4]>> = pipeline.query_async(&mut con).await?;

                let elapsed = Instant::now().duration_since(start_time);

                let results: Vec<Vec<u32>> = results
                    .into_iter()
                    .map(|r| r.into_iter().map(u32::from_le_bytes).collect())
                    .collect();

                let mut recalled = 0usize;
                for (pipeline_idx, result) in results.iter().enumerate() {
                    let id = ((start_q + batch_idx * pipeline_size + pipeline_idx) % queries.len())
                        as u32;
                    let all_true = truth.get(&id).unwrap();
                    let last_distance = all_true[k - 1].1;
                    let true_set: HashSet<_> = HashSet::from_iter(
                        all_true
                            .iter()
                            .copied()
                            .filter(|(_, d)| *d <= last_distance)
                            .map(|(id, _)| id),
                    );
                    let count = result
                        .iter()
                        .filter(|cand| true_set.contains(cand))
                        .count()
                        .min(k);

                    recalled += count;
                }
                stats.push((pipeline_size, recalled, k * pipeline_size, elapsed));

                tx.send(pipeline_size).await?;
            }

            Ok(stats)
        });
    }

    drop(tx);

    let mut total_searches = 0usize;
    let mut total_recalled = 0usize;
    let mut total_candidates = 0usize;
    let mut latencies = Vec::new();

    for result in tasks.join_all().await {
        let stats = result?;
        for (searches, recalled, candidates, elapsed) in stats {
            total_searches += searches;
            total_recalled += recalled;
            total_candidates += candidates;
            latencies.push(elapsed.as_micros() as f64 / args.pipeline_size as f64);
        }
    }

    let total_elapsed = time_start.elapsed().as_secs_f64();

    progress_task.await??;

    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;

    println!("RESULTS ({total_searches} queries finished in {total_elapsed:0.2}s):");
    println!("        qps: {:0.2}", total_searches as f64 / total_elapsed);
    println!(
        "     recall: {:0.2}%",
        100.0 * total_recalled as f64 / total_candidates as f64
    );
    println!("    latency: {avg_latency:0.2}us");

    Ok(())
}

async fn run<T: Element>(
    config: &Config,
    opts: &Options,
    args: &RunArgs,
    infos: Vec<redis::ConnectionInfo>,
    cred: Option<ExpiringCredential>,
) -> Result<()> {
    let book = Runbook::from_path(&args.runbook)?;
    let cat = Catalog::load_directory(&args.catalog, config.dataset_search_paths.as_deref())?;

    let parallelism = args.tasks.unwrap_or(thread::available_parallelism()?.get());
    let client = redis::Client::open(infos[0].clone())?;

    let filter = if !args.include.is_empty() || !args.exclude.is_empty() {
        let mut filter = Filter::default();
        for included in &args.include {
            filter.include(included);
        }
        for excluded in &args.exclude {
            filter.exclude(excluded);
        }
        Some(filter)
    } else {
        None
    };

    // Execute the runbook
    let driver = Garnet::<T>::new(
        client,
        cred,
        args.set.clone(),
        args.pipeline_size,
        parallelism,
        opts.quantizer.data_type(),
        args.degree,
        args.l_build,
        args.l_search,
        opts.quantizer,
    );
    let runner = Runner::new(driver);
    runner
        .run(
            &book,
            &cat,
            &args.report_path,
            args.k,
            args.n,
            filter,
            args,
            opts,
        )
        .await?;

    Ok(())
}
