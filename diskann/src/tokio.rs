/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

pub trait Spawn {
    fn spawn<F>(&self, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static;
}

impl Spawn for tokio::runtime::Handle {
    fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static,
    {
        tokio::runtime::Handle::spawn(self, future)
    }
}

impl Spawn for tokio::runtime::Runtime {
    fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future<Output: Send + 'static> + Send + 'static,
    {
        tokio::runtime::Runtime::spawn(self, future)
    }
}
