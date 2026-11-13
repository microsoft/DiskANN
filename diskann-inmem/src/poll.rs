/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

/// Call [`std::future::Future::poll`] once on `future` and return the result.
///
/// This can be used within this crate since the in-memory index is entirely synchronous
/// and `diskann` does not introduce its own asynchronisity.
///
/// # Panics
///
/// Panics if the single poll does not return [`std::task::Poll::Ready`].
pub(crate) fn poll<F>(future: F) -> F::Output
where
    F: std::future::Future,
{
    Poll::new(future).poll()
}

/// A wrapper for running a synchronous [`std::future::Future`].
#[derive(Debug)]
struct Poll<F>(F);

impl<F> Poll<F> {
    /// Construct a new [`Poll`] around `future`.
    fn new(future: F) -> Self {
        Self(future)
    }

    /// Call [`std::future::Future`] once on the contained future and return the result.
    ///
    /// # Panics
    ///
    /// Panics if the single poll does not return [`std::task::Poll::Ready`].
    #[expect(
        clippy::panic,
        reason = "we only pass synchronous futures to this internal method"
    )]
    fn poll(self) -> F::Output
    where
        F: std::future::Future,
    {
        let f = std::pin::pin!(self.into_inner());

        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::future::Future::poll(f, &mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("a non-synchronous future was polled"),
        }
    }

    /// Consume `self,` returning the inner future directly.
    fn into_inner(self) -> F {
        self.0
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_ready_output() {
        let value = String::from("ready");
        let mut polled = false;

        let output = poll(async {
            polled = true;
            value.as_str()
        });

        assert_eq!(output, "ready");
        assert!(polled);
    }

    #[test]
    #[should_panic(expected = "a non-synchronous future was polled")]
    fn panics_on_pending_without_repolling() {
        let mut polls = 0;

        poll(std::future::poll_fn(|_| {
            polls += 1;
            assert_eq!(polls, 1, "future must only be polled once");
            std::task::Poll::<()>::Pending
        }));
    }
}
