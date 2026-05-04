/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! The [`ObjectPool<C>`] class is a thread-safe queue that allows concurrent access and is
//! used for reusing allocated objects.
//!
//! # How this works
//!
//! The Pool struct can contain objects for which there is at least one implemention of the
//! [`AsPooled`] or [`TryAsPooled`] trait.
//!
//! The [`AsPooled`] trait is used for types that can be pooled (i.e, created or modified)
//! without supporting failures, while the [`TryAsPooled`] trait is used for types that can
//! where creation or modification is fallible.

use std::{
    collections::VecDeque,
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

/// A thread-safe queue that allows concurrent access and is used for pooling items.
#[derive(Debug)]
pub struct ObjectPool<T> {
    /// Queue of stored object.
    queue: Mutex<VecDeque<T>>,

    /// Maximum capacity of the pool. If pool is over maximum capacity, newly items will not
    /// be pushed back to the pool.
    capacity: Option<usize>,
}

impl<T> ObjectPool<T> {
    /// Create an object pool consisting of `initial_size` object initialized using
    /// [`AsPooled::create`].
    ///
    /// The argument `capacity` can be provided to place an upper bound on the number of
    /// objects in the pool. If `initial_size > capacity`, the pool will be created with
    /// excess items.
    pub fn new<A>(args: A, initial_size: usize, capacity: Option<usize>) -> Self
    where
        T: AsPooled<A>,
        A: Clone,
    {
        let queue = (0..initial_size).map(|_| T::create(args.clone())).collect();

        Self {
            queue: Mutex::new(queue),
            capacity,
        }
    }

    /// Create an object pool consisting of `initial_size` object initialized using
    /// [`TryAsPooled::try_create`].
    ///
    /// The argument `capacity` can be provided to place an upper bound on the number of
    /// objects in the pool. If `initial_size > capacity`, the pool will be created with
    /// excess items.
    pub fn try_new<A>(
        args: A,
        initial_size: usize,
        capacity: Option<usize>,
    ) -> Result<Self, T::Error>
    where
        T: TryAsPooled<A>,
        A: Clone,
    {
        let queue = (0..initial_size)
            .map(|_| T::try_create(args.clone()))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            queue: Mutex::new(queue),
            capacity,
        })
    }

    /// Try to retrieve an object from the pool. If the pool is non-empty, then invoke
    /// [`TryAsPooled::try_modify`] on the object with the given arguments. Otherwise, invoke
    /// [`TryAsPooled::try_create`] to create a new object.
    ///
    /// If `try_modify` fails, the potentially modified object will not be returned to the
    /// queue and will instead be dropped.
    pub fn try_get_ref<A>(&self, args: A) -> Result<PooledRef<'_, T>, T::Error>
    where
        T: TryAsPooled<A>,
    {
        let item = self.try_get_or_create(args)?;
        Ok(PooledRef {
            item: ManuallyDrop::new(item),
            parent: self,
        })
    }

    /// Try to retrieve an object from the pool. If the pool is non-empty, then invoke
    /// [`AsPooled::modify`] on the object with the given arguments. Otherwise, invoke
    /// [`AsPooled::create`] to create a new object.
    ///
    /// Unlike [`ObjectPool::try_get_ref`], this interface should be infallible.
    ///
    /// If `modify` panics, the potentially modified object will not be returned to the
    /// queue and will instead be dropped.
    pub fn get_ref<A>(&self, args: A) -> PooledRef<'_, T>
    where
        T: AsPooled<A>,
    {
        let item = self.get_or_create(args);
        PooledRef {
            item: ManuallyDrop::new(item),
            parent: self,
        }
    }

    /// Try to retrieve an object from the pool. If the pool is non-empty, then invoke
    /// [`TryAsPooled::modify`] on the object with the given arguments. Otherwise, invoke
    /// [`TryAsPooled::create`] to create a new object.
    ///
    /// If `try_modify` fails, the potentially modified object will not be returned to the
    /// queue and will instead be dropped.
    pub fn try_get<A>(self: &Arc<Self>, args: A) -> Result<PooledArc<T>, T::Error>
    where
        T: TryAsPooled<A>,
    {
        let item = self.try_get_or_create(args)?;
        Ok(PooledArc {
            item: ManuallyDrop::new(item),
            parent: self.clone(),
        })
    }

    /// Try to retrieve an object from the pool. If the pool is non-empty, then invoke
    /// [`AsPooled::modify`] on the object with the given arguments. Otherwise, invoke
    /// [`AsPooled::create`] to create a new object.
    ///
    /// Unlike [`ObjectPool::try_get`], this interface should be infallible.
    ///
    /// If `modify` panics, the potentially modified object will not be returned to the
    /// queue and will instead be dropped.
    pub fn get<A>(self: &Arc<Self>, args: A) -> PooledArc<T>
    where
        T: AsPooled<A>,
    {
        let item = self.get_or_create(args);
        PooledArc {
            item: ManuallyDrop::new(item),
            parent: self.clone(),
        }
    }

    /// Return the number of items currently in the queue.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Return whether or not the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    //-----------------//
    // Private Methods //
    //-----------------//

    fn try_get_or_create<A>(&self, args: A) -> Result<T, T::Error>
    where
        T: TryAsPooled<A>,
    {
        // Important: This needs to be on its own line instead of in the `if let` because
        // the latter will extend the lifetime of the lock so it is held when `try_modify`
        // or `try_create` is called.
        let maybe = self.lock().pop_front();
        if let Some(mut item) = maybe {
            item.try_modify(args)?;
            Ok(item)
        } else {
            T::try_create(args)
        }
    }

    fn get_or_create<A>(&self, args: A) -> T
    where
        T: AsPooled<A>,
    {
        // Important: This needs to be on its own line instead of in the `if let` because the
        // latter will extend the lifetime of the lock so it is held when `modify` or
        // `create` is called.
        let maybe = self.lock().pop_front();
        if let Some(mut item) = maybe {
            item.modify(args);
            item
        } else {
            T::create(args)
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<T>> {
        match self.queue.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // We trust the implementation of `VecDeque` to keep itself in a consistent
                // state regardless of panics.
                //
                // We endeavor to only call non-panicking methods while holding the lock
                // anyways.
                //
                // In particular, this means that implementations of `AsPooled` and
                // `TryAsPooled` are allowed to panic, and we cannot call the associated
                // methods while holding the lock.
                self.queue.clear_poison();
                poisoned.into_inner()
            }
        }
    }
}

/// Attempt to retrieve a pooled object from an [`ObjectPool`], creating a new one if the
/// queue is empty.
///
/// The goal of this trait is to modify an existing object if available such that it is
/// indistinguishable semantically from a newly created object, allowing user code to be
/// agnostic of the provenance of the created object.
pub trait TryAsPooled<A>
where
    Self: Sized,
{
    /// Any error that can occur during creation or modification.
    type Error;

    /// Create an instance of `Self` from the argument types.
    fn try_create(args: A) -> Result<Self, Self::Error>;

    /// Modify an existing object so it behaves semantically identical to an object
    /// constructed using [`try_create`].
    ///
    /// This is often trickier to achieve than first anticipated.
    ///
    /// Note that it's up to the user to decide what "semantically identical" means. For
    /// pooled objects like hash tables, the underlying capacity is not necessarily part of
    /// the identity of an object - but there are contexts where it matters.
    fn try_modify(&mut self, args: A) -> Result<(), Self::Error>;
}

/// Retrieve a pooled object from an [`ObjectPool`], creating a new one if the queue is empty.
///
/// The goal of this trait is to modify an existing object if available such that it is
/// indistinguishable semantically from a newly created object, allowing user code to be
/// agnostic of the provenance of the created object.
pub trait AsPooled<A> {
    /// Create an instance of `Self` from the argument types.
    fn create(args: A) -> Self;

    /// Modify an existing object so it behaves semantically identical to an object
    /// constructed using [`create`].
    ///
    /// This is often trickier to achieve than first anticipated.
    ///
    /// Note that it's up to the user to decide what "semantically identical" means. For
    /// pooled objects like hash tables, the underlying capacity is not necessarily part of
    /// the identity of an object - but there are contexts where it matters.
    fn modify(&mut self, args: A);
}

/// An [`AsPooled`] initializer for `Vec<T>`.
///
/// Creates or modifies an existing array to be the specified length, without specifying
/// the contents of the returned array.
///
/// This lack of specification means that existing `Vec`s that are modified to be the
/// configured length will contain values from a previous run.
#[derive(Debug, Clone, Copy)]
pub struct Undef {
    pub len: usize,
}

impl Undef {
    /// Construct a new [`Defaulted`].
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl<T> AsPooled<Undef> for Vec<T>
where
    T: Default + Clone,
{
    fn create(undef: Undef) -> Self {
        vec![T::default(); undef.len]
    }

    fn modify(&mut self, undef: Undef) {
        self.resize(undef.len, T::default())
    }
}

/// [`PooledRef<'a, T>`] is a pooled object wrapper with a **reference** to its parent pool.
///
/// When dropped, the contained object will be returned to the pool if there is space.
#[derive(Debug)]
pub struct PooledRef<'a, T> {
    item: ManuallyDrop<T>,
    parent: &'a ObjectPool<T>,
}

impl<T> Drop for PooledRef<'_, T> {
    fn drop(&mut self) {
        let mut guard = self.parent.lock();
        if guard.len() < self.parent.capacity.unwrap_or(usize::MAX) {
            // SAFETY: We do not access self.item again after this.
            guard.push_back(unsafe { ManuallyDrop::take(&mut self.item) });
        } else {
            // NOTE: The implementation of `T::drop` could panic. Release the lock first.
            std::mem::drop(guard);

            // SAFETY: We do not access self.item again after this.
            unsafe { ManuallyDrop::drop(&mut self.item) };
        }
    }
}

impl<T> Deref for PooledRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T> DerefMut for PooledRef<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
    }
}

/// [`PooledArc<T>`] is a pooled object wrapper with an **Arc** to its parent pool.
///
/// When dropped, the contained object will be returned to the pool if there is space.
///
/// Unlike [`PooledRef`], this object is `'static` (it `T: 'static`).
#[derive(Debug)]
pub struct PooledArc<T> {
    item: ManuallyDrop<T>,
    parent: Arc<ObjectPool<T>>,
}

impl<T> Drop for PooledArc<T> {
    fn drop(&mut self) {
        let mut guard = self.parent.lock();
        if guard.len() < self.parent.capacity.unwrap_or(usize::MAX) {
            // SAFETY: We do not access self.item again after this.
            guard.push_back(unsafe { ManuallyDrop::take(&mut self.item) });
        } else {
            // NOTE: The implementation of `T::drop` could panic. Release the lock first.
            std::mem::drop(guard);

            // SAFETY: We do not access self.item again after this.
            unsafe { ManuallyDrop::drop(&mut self.item) };
        }
    }
}

impl<T> Deref for PooledArc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

impl<T> DerefMut for PooledArc<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.item
    }
}

/// [`PoolOption<T>`] is an enum that can be either a non-pooled item or a pooled item.
/// This is used to allow the user to choose between using a pooled item or opt out of pooling.
#[derive(Debug)]
pub enum PoolOption<T> {
    NonPooled(T),
    Pooled(PooledArc<T>),
}

impl<T> PoolOption<T> {
    pub fn non_pooled(item: T) -> Self {
        PoolOption::NonPooled(item)
    }

    pub fn try_non_pooled_create<A>(args: A) -> Result<Self, T::Error>
    where
        T: TryAsPooled<A>,
    {
        Ok(PoolOption::NonPooled(T::try_create(args)?))
    }

    pub fn non_pooled_create<A>(args: A) -> Self
    where
        T: AsPooled<A>,
    {
        PoolOption::NonPooled(T::create(args))
    }

    pub fn pooled<A>(pool: &Arc<ObjectPool<T>>, args: A) -> Self
    where
        T: AsPooled<A>,
    {
        PoolOption::Pooled(pool.get(args))
    }

    pub fn try_pooled<A>(pool: &Arc<ObjectPool<T>>, args: A) -> Result<Self, T::Error>
    where
        T: TryAsPooled<A>,
    {
        Ok(PoolOption::Pooled(pool.try_get(args)?))
    }

    pub fn is_pooled(&self) -> bool {
        matches!(self, PoolOption::Pooled(_))
    }

    pub fn is_non_pooled(&self) -> bool {
        matches!(self, PoolOption::NonPooled(_))
    }
}

impl<T> Deref for PoolOption<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        match self {
            PoolOption::NonPooled(item) => item,
            PoolOption::Pooled(item) => item,
        }
    }
}

impl<T> DerefMut for PoolOption<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            PoolOption::NonPooled(item) => item,
            PoolOption::Pooled(item) => item,
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestItem {
        value: Box<u32>,
        panic_on_drop: bool,
    }

    impl TestItem {
        fn new(value: u32) -> Self {
            Self {
                value: Box::new(value),
                panic_on_drop: false,
            }
        }
    }

    impl AsPooled<u32> for TestItem {
        fn create(value: u32) -> Self {
            TestItem::new(value)
        }

        fn modify(&mut self, value: u32) {
            *self.value = value;
            self.panic_on_drop = false;
        }
    }

    impl TryAsPooled<i32> for TestItem {
        type Error = ();

        fn try_create(value: i32) -> Result<Self, Self::Error> {
            match value.try_into() {
                Ok(v) => Ok(TestItem::new(v)),
                Err(_) => Err(()),
            }
        }

        fn try_modify(&mut self, value: i32) -> Result<(), Self::Error> {
            match value.try_into() {
                Ok(v) => {
                    *self.value = v;
                    self.panic_on_drop = false;
                    Ok(())
                }
                Err(_) => Err(()),
            }
        }
    }

    impl Drop for TestItem {
        fn drop(&mut self) {
            if self.panic_on_drop {
                panic!("panicking on drop");
            }
        }
    }

    // A struct that panics on API calls to ensure we avoid Mutex poisoning.
    struct TestPanic;

    impl AsPooled<TestPanic> for TestItem {
        fn create(_: TestPanic) -> Self {
            panic!("panicking on create")
        }

        fn modify(&mut self, _: TestPanic) {
            panic!("panicking on modify")
        }
    }

    impl TryAsPooled<TestPanic> for TestItem {
        type Error = ();

        fn try_create(_: TestPanic) -> Result<Self, Self::Error> {
            panic!("panicking on try_create")
        }

        fn try_modify(&mut self, _: TestPanic) -> Result<(), Self::Error> {
            panic!("panicking on try_modify")
        }
    }

    #[test]
    fn test_pool_basic_tests() {
        let pool = ObjectPool::<TestItem>::new(42, 2, None);
        assert_eq!(pool.len(), 2);

        let item1 = pool.get_ref(100);
        assert_eq!(*item1.value, 100);
        assert_eq!(pool.len(), 1);

        let item2 = pool.get_ref(200);
        assert_eq!(*item2.value, 200);
        assert_eq!(pool.len(), 0);

        let item = pool.get_ref(300);
        assert_eq!(*item.value, 300);
        assert_eq!(pool.len(), 0);
        {
            let item = pool.get_ref(400);
            assert_eq!(*item.value, 400);
            assert_eq!(pool.len(), 0);
        }
        assert_eq!(pool.len(), 1); // Pooled item is pushed back to the pool
        {
            let item_a = pool.get_ref(500);
            assert_eq!(*item_a.value, 500);
            assert_eq!(pool.len(), 0); // new item not yet returned to pool
            let item_b = pool.get_ref(600);
            assert_eq!(*item_b.value, 600);
            assert_eq!(pool.len(), 0); // another new item not yet returned to pool
        }
        assert_eq!(pool.len(), 2); // Both Pooled items are pushed back to the pool

        let pool = ObjectPool::<TestItem>::new(42, 1, None);
        let item = pool.get_ref(100);
        assert_eq!(*item.value, 100);
    }

    #[test]
    fn test_pool_basic_tests_with_try() {
        // Create a pool with negative initial size to test error case
        let pool_result = ObjectPool::<TestItem>::try_new(-1, 2, Some(100));
        assert!(
            pool_result.is_err(),
            "Pool creation should fail with negative args"
        );

        let pool = ObjectPool::<TestItem>::try_new(42, 2, None).unwrap();
        assert_eq!(pool.len(), 2);
        let item1 = pool.try_get_ref(100).unwrap();
        assert_eq!(*item1.value, 100);
        assert_eq!(pool.len(), 1);
        let item2 = pool.try_get_ref(200).unwrap();
        assert_eq!(*item2.value, 200);
        assert_eq!(pool.len(), 0);
        let item = pool.try_get_ref(300).unwrap();
        assert_eq!(*item.value, 300);
        assert_eq!(pool.len(), 0);
        {
            let item = pool.try_get_ref(400).unwrap();
            assert_eq!(*item.value, 400);
            assert_eq!(pool.len(), 0);
        }
        assert_eq!(pool.len(), 1); // Pooled item is pushed back to the pool
        {
            let item_a = pool.try_get_ref(500).unwrap();
            assert_eq!(*item_a.value, 500);
            assert_eq!(pool.len(), 0); // new item not yet returned to pool
            let item_b = pool.try_get_ref(600).unwrap();
            assert_eq!(*item_b.value, 600);
            assert_eq!(pool.len(), 0); // another new item not yet returned to pool
        }
        assert_eq!(pool.len(), 2); // Both Pooled items are pushed back to the pool

        let pool = ObjectPool::<TestItem>::try_new(42, 1, Some(100)).unwrap();
        let item = pool.try_get_ref(100).unwrap();
        assert_eq!(*item.value, 100);
    }

    #[test]
    fn test_pool_with_arc() {
        let pool = &Arc::new(ObjectPool::<TestItem>::new(42, 1, None));
        let item = pool.get(100);
        assert_eq!(*item.value, 100);
        assert_eq!(pool.len(), 0);

        let item = pool.get(200);
        assert_eq!(*item.value, 200);
        assert_eq!(pool.len(), 0);
        {
            let item = pool.get(400);
            assert_eq!(*item.value, 400);
            assert_eq!(pool.len(), 0);
        }
        assert_eq!(pool.len(), 1); // Pooled item is pushed back to the pool
        let item = pool.try_get_ref(400).unwrap();
        assert_eq!(*item.value, 400);
        assert_eq!(pool.len(), 0); // new item not yet returned to pool
        let item = pool.try_get(500).unwrap();
        assert_eq!(*item.value, 500);
    }

    #[test]
    fn test_pool_max_capacity_ref() {
        let pool = ObjectPool::<TestItem>::new(42, 1, Some(1));
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert_eq!(pool.len(), pool.capacity.unwrap()); // size is at max_capacity
        {
            let item = pool.get_ref(100);
            assert_eq!(pool.len(), 0); // item is removed from the pool
            assert!(pool.is_empty());
            assert!(pool.len() < pool.capacity.unwrap()); // size is less than max_capacity
            assert_eq!(*item.value, 100);
        }
        assert_eq!(pool.len(), 1); // item is pushed back to the pool
        assert_eq!(pool.len(), pool.capacity.unwrap()); // size is at max_capacity
        {
            let item1 = pool.get_ref(100);
            assert_eq!(pool.len(), 0); // item is not pushed back to the pool
            let item2 = pool.get_ref(200);
            assert_eq!(pool.len(), 0); // item is not pushed back to the pool
            let item3 = pool.get_ref(300);
            assert_eq!(pool.len(), 0); // item is not pushed back to the pool
            assert!(*item1.value == 100 && *item2.value == 200 && *item3.value == 300);
            // all items are alive
        }
        assert_eq!(pool.len(), pool.capacity.unwrap()); // max_capacity is not exceeded
        assert_eq!(pool.len(), 1); // at most max_capacity items are in the pool
    }

    #[test]
    fn test_pool_max_capacity_pooled_item() {
        let pool = &Arc::new(ObjectPool::<TestItem>::new(42, 1, Some(1)));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.len(), pool.capacity.unwrap()); // size is at max_capacity
        {
            let item = pool.get(100);
            assert_eq!(pool.len(), 0); // item is removed from the pool
            assert!(pool.len() < pool.capacity.unwrap()); // size is less than max_capacity
            assert_eq!(*item.value, 100);
        }
        assert_eq!(pool.len(), 1); // item is pushed back to the pool
        assert_eq!(pool.len(), pool.capacity.unwrap()); // size is at max_capacity
        {
            let item1 = pool.get(100);
            assert_eq!(pool.len(), 0); // item is not pushed back to the pool
            let item2 = pool.get(200);
            assert_eq!(pool.len(), 0); // item is not pushed back to the pool
            let item3 = pool.get(300);
            assert_eq!(pool.len(), 0); // item is not pushed back to the pool
            assert!(*item1.value == 100 && *item2.value == 200 && *item3.value == 300);
            // all items are alive
        }
        assert_eq!(pool.len(), pool.capacity.unwrap()); // max_capacity is not exceeded
        assert_eq!(pool.len(), 1); // at most max_capacity items are in the pool
    }

    #[test]
    fn test_pool_options() {
        // Test non_pooled and non_pooled_create
        let item = PoolOption::non_pooled(TestItem::new(42));
        assert_eq!(*item.value, 42);
        assert!(item.is_non_pooled());
        assert!(!item.is_pooled());

        let item = PoolOption::<TestItem>::non_pooled_create(100);
        assert_eq!(*item.value, 100);
        assert!(item.is_non_pooled());

        // Test pooled
        let pool = Arc::new(ObjectPool::<TestItem>::new(42, 1, None));
        let item = PoolOption::pooled(&pool, 100);
        assert_eq!(*item.value, 100);
        assert!(item.is_pooled());
        assert!(!item.is_non_pooled());

        // Test try_pooled and try_non_pooled_create
        let item = PoolOption::try_pooled(&pool, 100).unwrap();
        assert_eq!(*item.value, 100);
        assert!(item.is_pooled());
        let item = PoolOption::<TestItem>::try_non_pooled_create(200).unwrap();
        assert_eq!(*item.value, 200);
        assert!(item.is_non_pooled());
        assert!(!item.is_pooled());
        let item_result = PoolOption::<TestItem>::try_non_pooled_create(-200);
        assert!(
            item_result.is_err(),
            "Creating non-pooled item with negative args should fail"
        );
    }

    #[test]
    fn test_pool_ref_deref_mut() {
        // Test PooledRef deref and deref_mut
        let pool = ObjectPool::<TestItem>::new(42, 1, None);
        let item = pool.get_ref(100);

        assert_eq!(*item.value, 100);
        let mut item = pool.get_ref(100);
        *item.value = 200;
        assert_eq!(*item.value, 200);

        let item_ref: &TestItem = &item;
        assert_eq!(*item_ref.value, 200);

        let item_ref_mut: &mut TestItem = &mut item;
        assert_eq!(*item_ref_mut.value, 200);

        *item_ref_mut.value = 300;
        assert_eq!(*item_ref_mut.value, 300);

        // Test PooledArc deref and deref_mut
        let pool = &Arc::new(ObjectPool::<TestItem>::new(42, 1, None));
        let mut item = pool.get_ref(100);
        assert_eq!(*item.value, 100);

        *item.value = 200;
        assert_eq!(*item.value, 200);

        // Test PoolOption deref and deref_mut for both pooled and non-pooled items
        let pool = Arc::new(ObjectPool::<TestItem>::new(42, 1, None));
        let mut item = PoolOption::pooled(&pool, 100);
        assert_eq!(*item.value, 100);

        *item.value = 200;
        assert_eq!(*item.value, 200);

        let mut item = PoolOption::non_pooled(TestItem::new(42));
        assert_eq!(*item.value, 42);

        *item.value = 100;
        assert_eq!(*item.value, 100);
    }

    //------------------//
    // Panic Resiliance //
    //------------------//

    // Here - we test that panics during:
    //
    // * Object Creation
    // * Object Modification
    // * Object destruction (when the queue is full)
    //
    // Do not poison the central mutex.
    //
    // The goal is to test code around anything that locks the mutex to ensure it only
    // invokes non-panicking code, and any code that can potentially panic is run outside
    // of the lock.

    // Check that the panicking payload is castable to a `&'static str` and that it
    // contains `msg`.
    fn check_error(err: &dyn std::any::Any, contains: &str) {
        match err.downcast_ref::<&'static str>() {
            Some(msg) => assert!(
                msg.contains(contains),
                "failed: message \"{}\" does not contain \"{}\"",
                msg,
                contains
            ),
            None => panic!("incorrect downcast type"),
        }
    }

    #[test]
    fn test_panic_during_create() {
        let pool = ObjectPool::<TestItem>::new(0u32, 0, Some(1));

        // Panic during `create`.
        let err = std::panic::catch_unwind(|| {
            let _ = pool.get_ref(TestPanic);
        })
        .unwrap_err();

        check_error(&*err, "panicking on create");

        assert!(
            !pool.queue.is_poisoned(),
            "lock should be released while calling trait implementations"
        );

        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_panic_during_try_create() {
        let pool = ObjectPool::<TestItem>::new(0u32, 0, Some(1));

        // Panic during `try_create`.
        let err = std::panic::catch_unwind(|| {
            let _ = pool.try_get_ref(TestPanic);
        })
        .unwrap_err();

        check_error(&*err, "panicking on try_create");

        assert!(
            !pool.queue.is_poisoned(),
            "lock should be released while calling trait implementations"
        );

        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_panic_during_modify() {
        let pool = ObjectPool::<TestItem>::new(0u32, 0, Some(1));

        // Append a new item into the pool so it is full.
        let _ = pool.get_ref(0u32);
        assert_eq!(pool.len(), 1);

        let err = std::panic::catch_unwind(|| {
            let _ = pool.get_ref(TestPanic);
        })
        .unwrap_err();

        check_error(&*err, "panicking on modify");

        assert!(
            !pool.queue.is_poisoned(),
            "lock should be released while calling trait implementations"
        );

        assert_eq!(
            pool.len(),
            0,
            "we should not return a potentially torn object to the pool"
        );
    }

    #[test]
    fn test_panic_during_try_modify() {
        let pool = ObjectPool::<TestItem>::new(0u32, 0, Some(1));

        // Append a new item into the pool so it is full.
        let _ = pool.get_ref(0u32);
        assert_eq!(pool.len(), 1);

        let err = std::panic::catch_unwind(|| {
            let _ = pool.try_get_ref(TestPanic);
        })
        .unwrap_err();

        check_error(&*err, "panicking on try_modify");

        assert!(
            !pool.queue.is_poisoned(),
            "lock should be released while calling trait implementations"
        );

        assert_eq!(
            pool.len(),
            0,
            "we should not return a potentially torn object to the pool"
        );
    }

    // Panic on full drop - `ref` interface.
    #[test]
    fn test_panic_during_drop_ref() {
        let pool = ObjectPool::<TestItem>::new(0u32, 0, Some(1));

        // This sequence ensures that when we try to put the object behind `a` back
        // into the queue that it is dropped since the queue is full.
        let mut a = pool.get_ref(0u32);
        let _ = pool.get_ref(1u32);
        assert_eq!(pool.len(), 1);

        a.panic_on_drop = true;
        let err = std::panic::catch_unwind(move || std::mem::drop(a)).unwrap_err();
        check_error(&*err, "panicking on drop");

        assert!(
            !pool.queue.is_poisoned(),
            "lock should be released while calling object drop"
        );
        assert_eq!(pool.len(), 1);
    }

    // Panic on full drop - `Arc` interface.
    #[test]
    fn test_panic_during_drop_arc() {
        let pool = Arc::new(ObjectPool::<TestItem>::new(0u32, 0, Some(1)));

        // This sequence ensures that when we try to put the object behind `a` back
        // into the queue that it is dropped since the queue is full.
        let mut a = pool.get(0u32);
        let _ = pool.get(1u32);
        assert_eq!(pool.len(), 1);

        a.panic_on_drop = true;
        let err = std::panic::catch_unwind(move || std::mem::drop(a)).unwrap_err();
        check_error(&*err, "panicking on drop");

        assert!(
            !pool.queue.is_poisoned(),
            "lock should be released while calling object drop"
        );
        assert_eq!(pool.len(), 1);
    }

    // This test uses private member access to induce a poisoned state in the mutex.
    #[test]
    fn test_panic_recovery() {
        let pool = ObjectPool::<TestItem>::new(0u32, 1, Some(1));

        let err = std::panic::catch_unwind(|| {
            let _guard = pool.queue.lock();
            panic!("yeet");
        })
        .unwrap_err();

        check_error(&*err, "yeet");

        assert!(pool.queue.is_poisoned());

        let _ = pool.get_ref(1u32);
        assert!(!pool.queue.is_poisoned(), "poison should be cleared");
    }

    //-------//
    // Undef //
    //-------//

    #[test]
    fn test_undef() {
        let mut x: Vec<f32> = Vec::<f32>::create(Undef::new(10));
        assert_eq!(x.len(), 10);

        x.modify(Undef::new(0));
        assert_eq!(x.len(), 0);

        x.modify(Undef::new(20));
        assert_eq!(x.len(), 20);
    }
}
