use anyhow::Result;
use azure_core::{credentials::TokenCredential, time::OffsetDateTime};
use azure_identity::AzureCliCredential;
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use loader::DatasetLoader;
use redis::{AsyncTypedCommands, Pipeline, ToRedisArgs, aio::MultiplexedConnection};
use serde::Deserialize;
use std::{
    collections::HashSet,
    path::PathBuf,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use tokio::{
    fs::File,
    io::AsyncReadExt,
    sync::mpsc,
    task::{JoinHandle, JoinSet},
};

mod loader;

const DEFAULT_PORT: u16 = 6379;

trait Element: bytemuck::Pod + std::fmt::Debug + Send + Sync + 'static {}

impl Element for u8 {}
impl Element for f32 {}

#[derive(Deserialize)]
struct Config {
    ips: Vec<String>,
    port: Option<u16>,
    secure: bool,
    scope: String,
    username: Option<String>,
}

#[derive(Parser)]
struct Options {
    /// Path to config file
    #[arg(short = 'C', long, value_name = "CONFIG_FILE")]
    config: PathBuf,

    /// Maximum number of threads for the runtime
    #[arg(short, long)]
    threads: Option<usize>,

    /// Data type for vector elements
    #[arg(long)]
    data_type: DataType,

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

    /// Paths to base vectors
    base_paths: Vec<PathBuf>,
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

    /// Number of search results to return
    #[arg(short, long, default_value = "10")]
    k: usize,

    /// Number of ground vectors to consider
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

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum DataType {
    Uint8,
    Float32,
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

    async fn refresh_if_needed(mut self, con: &mut MultiplexedConnection) -> Result<Self> {
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

    let (password, expires) = if config.username.is_some() {
        let credentials = AzureCliCredential::new(None)?;
        let res = credentials.get_token(&[&config.scope], None).await?;
        (Some(res.token.secret().to_string()), Some(res.expires_on))
    } else {
        (None, None)
    };

    let mut infos = Vec::new();
    for addr in addrs.into_iter() {
        let info = redis::ConnectionInfo {
            addr,
            redis: redis::RedisConnectionInfo {
                username: config.username.clone(),
                password: password.clone(),
                ..Default::default()
            },
        };
        infos.push(info);
    }

    let cred = if config.username.is_some() {
        Some(ExpiringCredential::new(
            config.scope.clone(),
            config.username.clone().unwrap(),
            AzureCliCredential::new(None)?,
            expires.unwrap(),
        ))
    } else {
        None
    };

    match opts.data_type {
        DataType::Uint8 => dispatch::<u8>(&opts.command, &opts, infos, cred).await,
        DataType::Float32 => dispatch::<f32>(&opts.command, &opts, infos, cred).await,
    }
}

async fn dispatch<T: Element>(
    command: &Commands,
    opts: &Options,
    infos: Vec<redis::ConnectionInfo>,
    cred: Option<ExpiringCredential>,
) -> Result<()> {
    match command {
        Commands::Ping => ping::<T>(infos[0].clone()).await?,
        Commands::Ingest(args) => ingest::<T>(opts, args, infos[0].clone(), cred).await?,
        Commands::Delete(args) => delete::<T>(args, infos[0].clone()).await?,
        Commands::Query(args) => query::<T>(opts, args, infos, cred).await?,
    }

    Ok(())
}

async fn ping<T: Element>(info: redis::ConnectionInfo) -> Result<()> {
    let client = redis::Client::open(info).unwrap();
    let mut con = client.get_multiplexed_async_connection().await?;

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
    let ds = if let Some(dim) = args.no_header_with_dim {
        DatasetLoader::new_with_headerless_dim(&args.base_paths, Some(dim)).await?
    } else {
        DatasetLoader::new(&args.base_paths).await?
    };

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
        let mut con = client.get_multiplexed_async_connection().await?;
        let ds = ds.clone();
        let pipeline_size = args.pipeline_size;
        let vset = vset.clone();
        let tx = tx.clone();
        let limit = args.limit.unwrap_or(ds.len());
        let l_build = args.l_build;
        let degree = args.degree;
        let mut cred = cred.clone();
        let data_type = opts.data_type;

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
                                pipeline.arg(b"XB8");
                            }
                            DataType::Float32 => {
                                pipeline.arg(b"FP32");
                            }
                        }

                        pipeline
                            .arg(bytemuck::cast_slice::<_, u8>(&buf[buf_start..buf_end]))
                            .arg(element);

                        match data_type {
                            DataType::Uint8 => {
                                pipeline.arg(b"XPREQ8");
                            }
                            DataType::Float32 => {
                                pipeline.arg(b"NOQUANT");
                            }
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

async fn delete<T: Element>(args: &DeleteArgs, info: redis::ConnectionInfo) -> Result<()> {
    let client = redis::Client::open(info).unwrap();
    let mut con = client.get_multiplexed_async_connection().await?;

    con.del(&args.set).await?;
    con.flushdb().await?;
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
    let truth = loader::load_groundtruth(&args.gt_path).await?;
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
        let mut con = client.get_multiplexed_async_connection().await?;
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
        let data_type = opts.data_type;

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
                        DataType::Uint8 => pipeline.arg(b"XB8"),
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
