/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{marker::PhantomData, time::Instant};

use redis::{AsyncCommands, Pipeline};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::{
    DataType, DistanceMetric, Element, ExpiringCredential, ExpiringCredentialError, Quantizer,
    VectorId,
    dataset::RowBuf,
    driver::{ControllerError, Driver, SearchResults, Timings},
};

#[derive(Debug, Error)]
pub enum GarnetError {
    #[error("redis error: {0}")]
    Redis(#[from] redis::RedisError),
    #[error("credential error: {0}")]
    Credential(#[from] ExpiringCredentialError),
    #[error("controller error: {0}")]
    Controller(#[from] ControllerError),
    #[error("replace tags/ids mismatch (ids: {0}, tags: {1})")]
    ReplaceMismatch(usize, usize),
}

pub struct Garnet<T: bytemuck::Pod + Default + Send + Sync> {
    client: redis::Client,
    cred: Option<ExpiringCredential>,

    vset: String,
    pipeline_size: usize,
    parallelism: usize,
    data_type: DataType,

    degree: usize,
    l_build: usize,
    l_search: usize,
    quantizer: Quantizer,

    _phantom: PhantomData<T>,
}

impl<T: bytemuck::Pod + Default + Send + Sync> Garnet<T> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: redis::Client,
        cred: Option<ExpiringCredential>,
        vset: String,
        pipeline_size: usize,
        parallelism: usize,
        data_type: DataType,
        degree: usize,
        l_build: usize,
        l_search: usize,
        quantizer: Quantizer,
    ) -> Self {
        Self {
            client,
            cred,
            vset,
            pipeline_size,
            parallelism,
            data_type,
            degree,
            l_build,
            l_search,
            quantizer,
            _phantom: PhantomData,
        }
    }
}

impl<T: Element> Driver for Garnet<T> {
    type Connection = redis::aio::MultiplexedConnection;
    type Error = GarnetError;
    type Data = T;

    fn name(&self) -> String {
        "Garnet".to_string()
    }

    fn parallelism(&self) -> usize {
        self.parallelism
    }

    async fn get_connection(&self) -> Result<Self::Connection, Self::Error> {
        Ok(self
            .client
            .get_multiplexed_async_connection_with_config(&crate::connection_config())
            .await?)
    }

    async fn prepare(&self, mut con: Self::Connection) -> Result<(), Self::Error> {
        let _: usize = con.del(self.vset.as_bytes()).await?;
        Ok(())
    }

    async fn finish(&self, mut con: Self::Connection) -> Result<(), Self::Error> {
        let _: usize = con.del(self.vset.as_bytes()).await?;
        Ok(())
    }

    async fn insert(
        &self,
        mut con: Self::Connection,
        metric: DistanceMetric,
        vectors: RowBuf<Self::Data>,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> Result<Timings, Self::Error> {
        let mut pipeline = Pipeline::with_capacity(self.pipeline_size);
        let mut cred = self.cred.clone();

        let start = vectors.start();
        let end = start + vectors.nrows();

        let mut timings = Vec::new();
        let mut id = start;
        while id < end {
            if let Some(c) = cred {
                cred = Some(c.refresh_if_needed(&mut con).await?);
            }

            while id < end {
                pipeline.clear();

                let count = (end - id).min(self.pipeline_size);
                for i in 0..count {
                    let element = VectorId((id + i) as u32);

                    pipeline.cmd("VADD").arg(&self.vset);

                    match self.data_type {
                        DataType::Float32 => {
                            pipeline.arg(b"FP32");
                        }
                        DataType::Int8 => {
                            pipeline.arg(b"XI8");
                        }
                        DataType::Uint8 => {
                            pipeline.arg(b"XU8");
                        }
                    }

                    pipeline
                        .arg(bytemuck::cast_slice::<T, u8>(vectors.row(id + i)))
                        .arg(element);

                    pipeline.arg(self.quantizer);

                    pipeline.arg(b"XDISTANCE_METRIC").arg(metric);

                    pipeline
                        .arg(b"EF")
                        .arg(self.l_build.to_string().as_bytes())
                        .arg(b"M")
                        .arg(self.degree.to_string().as_bytes());
                }

                id += count;

                let started = Instant::now();

                pipeline.exec_async(&mut con).await?;

                let duration = Instant::now().duration_since(started);

                timings.push((count, duration));

                // A closed channel just means nobody is collecting stats.
                let _ = reporter.send(count);
            }
        }

        Ok(timings)
    }

    async fn delete(
        &self,
        mut con: Self::Connection,
        start: usize,
        end: usize,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> Result<Timings, Self::Error> {
        let mut pipeline = Pipeline::with_capacity(self.pipeline_size);
        let mut cred = self.cred.clone();

        let mut timings = Vec::new();
        let mut id = start;
        while id < end {
            if let Some(c) = cred {
                cred = Some(c.refresh_if_needed(&mut con).await?);
            }

            while id < end {
                pipeline.clear();

                let count = (end - id).min(self.pipeline_size);
                for i in 0..count {
                    let element = VectorId((id + i) as u32);

                    pipeline.cmd("VREM").arg(&self.vset).arg(element);
                }

                id += count;

                let started = Instant::now();

                pipeline.exec_async(&mut con).await?;

                let duration = Instant::now().duration_since(started);

                timings.push((count, duration));

                // A closed channel just means nobody is collecting stats.
                let _ = reporter.send(count);
            }
        }

        Ok(timings)
    }

    async fn replace(
        &self,
        mut con: Self::Connection,
        metric: DistanceMetric,
        tags_start: usize,
        tags_end: usize,
        vectors: RowBuf<Self::Data>,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> Result<Timings, Self::Error> {
        let mut pipeline = Pipeline::with_capacity(self.pipeline_size);
        let mut cred = self.cred.clone();

        let start = vectors.start();
        let end = start + vectors.nrows();

        if vectors.nrows() != tags_end - tags_start {
            return Err(GarnetError::ReplaceMismatch(
                vectors.nrows(),
                tags_end - tags_start,
            ));
        }

        let mut timings = Vec::new();
        let mut id = start;
        let mut tag = tags_start;
        while id < end {
            if let Some(c) = cred {
                cred = Some(c.refresh_if_needed(&mut con).await?);
            }

            while id < end {
                pipeline.clear();

                let count = (end - id).min(self.pipeline_size);
                for i in 0..count {
                    let element = VectorId((tag + i) as u32);

                    pipeline.cmd("VADD").arg(&self.vset);

                    match self.data_type {
                        DataType::Float32 => {
                            pipeline.arg(b"FP32");
                        }
                        DataType::Int8 => {
                            pipeline.arg(b"XI8");
                        }
                        DataType::Uint8 => {
                            pipeline.arg(b"XU8");
                        }
                    }

                    pipeline
                        .arg(bytemuck::cast_slice::<T, u8>(vectors.row(id + i)))
                        .arg(element);

                    pipeline.arg(self.quantizer);

                    pipeline.arg(b"XDISTANCE_METRIC").arg(metric);

                    pipeline
                        .arg(b"EF")
                        .arg(self.l_build.to_string().as_bytes())
                        .arg(b"M")
                        .arg(self.degree.to_string().as_bytes());
                }

                id += count;
                tag += count;

                let started = Instant::now();

                pipeline.exec_async(&mut con).await?;

                let duration = Instant::now().duration_since(started);

                timings.push((count, duration));

                // A closed channel just means nobody is collecting stats.
                let _ = reporter.send(count);
            }
        }

        Ok(timings)
    }

    async fn search(
        &self,
        mut con: Self::Connection,
        queries: RowBuf<Self::Data>,
        recall_n: usize,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> Result<SearchResults, Self::Error> {
        let mut cred = self.cred.clone();

        let start = queries.start();
        let end = start + queries.nrows();

        let mut timings = Vec::new();
        let mut results = Vec::new();

        let mut id = start;
        while id < end {
            if let Some(c) = cred {
                cred = Some(c.refresh_if_needed(&mut con).await?);
            }

            let mut cmd = redis::cmd("VSIM");
            cmd.arg(&self.vset);

            match self.data_type {
                DataType::Float32 => {
                    cmd.arg(b"FP32");
                }
                DataType::Int8 => {
                    cmd.arg(b"XI8");
                }
                DataType::Uint8 => {
                    cmd.arg(b"XU8");
                }
            }

            cmd.arg(bytemuck::cast_slice::<T, u8>(queries.row(id)))
                .arg(b"COUNT")
                .arg(recall_n.to_string().as_bytes())
                .arg(b"EF")
                .arg(self.l_search.to_string().as_bytes());

            id += 1;

            let started = Instant::now();

            let all_results: Vec<[u8; 4]> = cmd.query_async(&mut con).await?;

            let duration = Instant::now().duration_since(started);

            timings.push((1, duration));
            results.push(
                all_results
                    .into_iter()
                    .map(|b| {
                        let mut id = 0u32;
                        bytemuck::bytes_of_mut(&mut id).copy_from_slice(&b);
                        id
                    })
                    .collect::<Vec<_>>(),
            );

            // A closed channel just means nobody is collecting stats.
            let _ = reporter.send(1);
        }

        Ok((timings, start, results))
    }
}
