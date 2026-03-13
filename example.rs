// Default post-process
#[derive(Debug, Clone, Copy)]
pub struct DefaultPostProcess;

pub trait DelegatePostProcess<Args...?> {
    type Delegate: DoesThings;
}

impl<T, Args...> SearchPostProcess<T, Args...> for DefaultPostProcess
where
    T: DelegatePostProcess<Args...?>
{
    fn post_process(args...) {
        T::Delegate::post_process(args...)
    }
}

// Apply the default post-process via the normal search API.
fn search<T>(
    dispatch: T,
    other_args...
)
where
    DefaultPostProcess: SearchPostProcess<OtherArgs...>
{
    search_with(dispatch, other_args..., DefaultPostProcess)
}

// Second API that allows for overriding the post-processor explicitly.
fn search_with<T, P>(
    dispatch: T,
    other_args...
    post_process: P
)
where
    P: SearchPostProcess<OtherArgs...>
{
    // Do the thing. The `Search` trait will always take a post-processor.
}

