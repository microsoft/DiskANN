/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::time::Duration;

use thiserror::Error;
use tokio::sync::mpsc;

use crate::{DistanceMetric, Element, dataset::RowBuf};

#[derive(Debug, Error)]
#[error("controller: {0}")]
pub struct ControllerError(pub Box<dyn std::error::Error + Send + Sync + 'static>);

pub type Timings = Vec<(usize, Duration)>;
pub type SearchResults = (Timings, usize, Vec<Vec<u32>>);

pub trait Driver {
    type Connection: Send;
    type Error: std::error::Error + Send + Sync + 'static + From<ControllerError>;
    type Data: Element + Send;

    fn name(&self) -> String;
    fn parallelism(&self) -> usize;
    fn get_connection(&self) -> impl Future<Output = Result<Self::Connection, Self::Error>> + Send;
    fn prepare(
        &self,
        con: Self::Connection,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;
    fn finish(&self, con: Self::Connection)
    -> impl Future<Output = Result<(), Self::Error>> + Send;

    fn insert(
        &self,
        con: Self::Connection,
        metric: DistanceMetric,
        vectors: RowBuf<Self::Data>,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> impl Future<Output = Result<Timings, Self::Error>> + Send;

    fn delete(
        &self,
        con: Self::Connection,
        start: usize,
        end: usize,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> impl Future<Output = Result<Timings, Self::Error>> + Send;

    fn replace(
        &self,
        con: Self::Connection,
        metric: DistanceMetric,
        tags_start: usize,
        tags_end: usize,
        vectors: RowBuf<Self::Data>,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> impl Future<Output = Result<Timings, Self::Error>> + Send;

    fn search(
        &self,
        con: Self::Connection,
        queries: RowBuf<Self::Data>,
        recall_n: usize,
        reporter: mpsc::UnboundedSender<usize>,
    ) -> impl Future<Output = Result<SearchResults, Self::Error>> + Send;
}
