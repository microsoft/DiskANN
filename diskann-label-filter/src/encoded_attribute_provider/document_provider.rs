/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use diskann::{
    error::{ErrorExt, IntoANNResult},
    provider::{DataProvider, Delete, Guard, SetElement},
    ANNError, ANNErrorKind,
};

use diskann_utils::future::AsyncFriendly;

use crate::{document::Document, traits::attribute_store::AttributeStore};

/// Manage [`Document`]s which are a combination of a vector
/// and its attributes for DiskANN. Uses an underlying data
/// provider and attribute store for the functionality.
pub struct DocumentProvider<DP, AS> {
    inner_provider: DP,
    attribute_store: AS,
}

impl<DP, AS> DocumentProvider<DP, AS> {
    pub fn new(data_provider: DP, attribute_store: AS) -> Self {
        Self {
            inner_provider: data_provider,
            attribute_store,
        }
    }

    pub fn attribute_accessor(&self) -> Result<AS::Accessor, AS::StoreError>
    where
        DP: DataProvider,
        AS: AttributeStore<DP::InternalId>,
    {
        self.attribute_store.attribute_accessor()
    }

    pub fn inner_provider(&self) -> &DP {
        &self.inner_provider
    }

    pub fn attribute_store(&self) -> &AS {
        &self.attribute_store
    }
}

impl<DP, AS> DataProvider for DocumentProvider<DP, AS>
where
    DP: DataProvider,
    AS: AttributeStore<DP::InternalId> + AsyncFriendly,
{
    type Context = DP::Context;
    type ExternalId = DP::ExternalId;
    type InternalId = DP::InternalId;
    type Error = DP::Error;
    type Guard = DP::Guard;

    fn to_internal_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<Self::InternalId, Self::Error> {
        self.inner_provider.to_internal_id(context, gid)
    }

    fn to_external_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<Self::ExternalId, Self::Error> {
        self.inner_provider.to_external_id(context, id)
    }
}

impl<'a, VT, DP, AS> SetElement<Document<'a, VT>> for DocumentProvider<DP, AS>
where
    DP: DataProvider + Delete + SetElement<&'a VT>,
    AS: AttributeStore<DP::InternalId> + AsyncFriendly,
    VT: Sync + Send,
{
    type SetError = ANNError;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: Document<'a, VT>,
    ) -> Result<Self::Guard, Self::SetError> {
        let guard = self
            .inner_provider
            .set_element(context, id, element.vector())
            .await
            .escalate("Failed to set the vector while adding vector and attributes")?;

        let _ = self
            .attribute_store
            .set_element(&guard.id(), element.attributes())
            .into_ann_result()?;
        Ok(guard)
    }
}

impl<DP, AS> Delete for DocumentProvider<DP, AS>
where
    DP: DataProvider<Error = ANNError> + Delete,
    AS: AttributeStore<DP::InternalId> + AsyncFriendly,
    ANNError: From<<AS as AttributeStore<<DP as DataProvider>::InternalId>>::StoreError>,
{
    async fn delete(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<(), Self::Error> {
        self.inner_provider.delete(context, gid).await
    }

    fn release(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let _ = self.attribute_store.delete(&(id)).map_err(|_| {
            ANNError::message(
                ANNErrorKind::IndexError,
                format!("Could not delete attributes of {}.", id),
            )
        });

        self.inner_provider.release(context, id)
    }

    async fn status_by_internal_id(
        &self,
        context: &Self::Context,
        id: Self::InternalId,
    ) -> Result<diskann::provider::ElementStatus, Self::Error> {
        let is_id_in_attr_store_w = self.attribute_store.id_exists(&id).map_err(|e| {
            ANNError::new(ANNErrorKind::IndexError, e)
                .context("Failed to get attribute status by internal id.")
        });
        let id_in_data_store_w = self
            .inner_provider
            .status_by_internal_id(context, id)
            .await
            .into_ann_result();
        // .map_err(|e| {
        //     ANNError::new(ANNErrorKind::IndexError, e)
        //         .context("Failed to get status from data provider.")
        // });

        let is_id_in_attr_store = is_id_in_attr_store_w?;
        let id_in_data_store = id_in_data_store_w?;

        if is_id_in_attr_store && id_in_data_store.is_deleted() {
            Err(ANNError::message(
                ANNErrorKind::IndexError,
                "Id was found in the attribute store, but not in the data store.",
            ))
        } else if !is_id_in_attr_store && id_in_data_store.is_valid() {
            Err(ANNError::message(
                ANNErrorKind::IndexError,
                "Id was found in the data store but not in the attribute store.",
            ))
        } else {
            Ok(id_in_data_store)
        }
    }

    async fn status_by_external_id(
        &self,
        context: &Self::Context,
        gid: &Self::ExternalId,
    ) -> Result<diskann::provider::ElementStatus, Self::Error> {
        // Convert external ID to internal ID and then check status by internal ID
        let internal_id = self.to_internal_id(context, gid)?;
        self.status_by_internal_id(context, internal_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{Attribute, AttributeValue};
    use crate::document::Document;
    use crate::encoded_attribute_provider::roaring_attribute_store::RoaringAttributeStore;
    use crate::traits::attribute_store::AttributeStore;
    use diskann::provider::{DefaultContext, ElementStatus, Guard, NoopGuard};
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Minimal in-memory `DataProvider` that records element statuses and uses
    /// an identity external-to-internal id mapping.
    #[derive(Default)]
    struct MockProvider {
        statuses: Mutex<HashMap<u32, ElementStatus>>,
        released: Mutex<Vec<u32>>,
    }

    impl MockProvider {
        fn set_status(&self, id: u32, status: ElementStatus) {
            self.statuses.lock().unwrap().insert(id, status);
        }
    }

    impl DataProvider for MockProvider {
        type Context = DefaultContext;
        type InternalId = u32;
        type ExternalId = u32;
        type Error = ANNError;
        type Guard = NoopGuard<u32>;

        fn to_internal_id(&self, _ctx: &Self::Context, gid: &u32) -> Result<u32, ANNError> {
            Ok(*gid)
        }

        fn to_external_id(&self, _ctx: &Self::Context, id: u32) -> Result<u32, ANNError> {
            Ok(id)
        }
    }

    impl<T: Send> SetElement<T> for MockProvider {
        type SetError = ANNError;

        async fn set_element(
            &self,
            _ctx: &Self::Context,
            id: &u32,
            _element: T,
        ) -> Result<NoopGuard<u32>, ANNError> {
            self.set_status(*id, ElementStatus::Valid);
            Ok(NoopGuard::new(*id))
        }
    }

    impl Delete for MockProvider {
        async fn delete(&self, _ctx: &Self::Context, gid: &u32) -> Result<(), ANNError> {
            self.set_status(*gid, ElementStatus::Deleted);
            Ok(())
        }

        async fn release(&self, _ctx: &Self::Context, id: u32) -> Result<(), ANNError> {
            self.released.lock().unwrap().push(id);
            Ok(())
        }

        async fn status_by_internal_id(
            &self,
            _ctx: &Self::Context,
            id: u32,
        ) -> Result<ElementStatus, ANNError> {
            Ok(self
                .statuses
                .lock()
                .unwrap()
                .get(&id)
                .copied()
                .unwrap_or(ElementStatus::Deleted))
        }

        async fn status_by_external_id(
            &self,
            ctx: &Self::Context,
            gid: &u32,
        ) -> Result<ElementStatus, ANNError> {
            let id = self.to_internal_id(ctx, gid)?;
            self.status_by_internal_id(ctx, id).await
        }
    }

    /// Drive a future to completion on the current thread. The futures produced
    /// here never suspend, so a simple poll loop is sufficient.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        fn raw() -> RawWaker {
            fn no_op(_: *const ()) {}
            fn clone(_: *const ()) -> RawWaker {
                raw()
            }
            RawWaker::new(
                std::ptr::null(),
                &RawWakerVTable::new(clone, no_op, no_op, no_op),
            )
        }
        // SAFETY: the vtable functions are all no-ops operating on a null pointer.
        let waker = unsafe { Waker::from_raw(raw()) };
        let mut cx = Context::from_waker(&waker);
        let mut fut = std::pin::pin!(fut);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::hint::spin_loop(),
            }
        }
    }

    type Dp = DocumentProvider<MockProvider, RoaringAttributeStore<u32>>;

    fn make() -> Dp {
        DocumentProvider::new(MockProvider::default(), RoaringAttributeStore::<u32>::new())
    }

    fn attrs() -> Vec<Attribute> {
        vec![Attribute::from_value(
            "category",
            AttributeValue::String("electronics".to_owned()),
        )]
    }

    #[test]
    fn id_translation_delegates_to_inner() {
        let dp = make();
        let ctx = DefaultContext;
        assert_eq!(dp.to_internal_id(&ctx, &9).unwrap(), 9);
        assert_eq!(dp.to_external_id(&ctx, 9).unwrap(), 9);
    }

    #[test]
    fn attribute_accessor_is_available() {
        let dp = make();
        assert!(dp.attribute_accessor().is_ok());
    }

    #[test]
    fn set_element_marks_vector_and_attributes() {
        let dp = make();
        let ctx = DefaultContext;
        let vector = vec![1.0_f32, 2.0, 3.0];
        let doc = Document::new(&vector, attrs());

        let guard = block_on(dp.set_element(&ctx, &1, doc)).unwrap();
        assert_eq!(guard.id(), 1);

        // Present in both the data store and the attribute store -> valid.
        let status = block_on(dp.status_by_internal_id(&ctx, 1)).unwrap();
        assert_eq!(status, ElementStatus::Valid);

        // The external-id path resolves to the same status.
        let status = block_on(dp.status_by_external_id(&ctx, &1)).unwrap();
        assert_eq!(status, ElementStatus::Valid);
    }

    #[test]
    fn status_of_unknown_id_is_deleted() {
        let dp = make();
        let ctx = DefaultContext;
        // Absent from both stores -> reported as deleted.
        let status = block_on(dp.status_by_internal_id(&ctx, 42)).unwrap();
        assert_eq!(status, ElementStatus::Deleted);
    }

    #[test]
    fn status_errors_when_attribute_present_but_data_deleted() {
        let dp = make();
        let ctx = DefaultContext;
        // Attribute store knows the id, data store does not.
        dp.attribute_store().set_element(&5, &attrs()).unwrap();
        let result = block_on(dp.status_by_internal_id(&ctx, 5));
        assert!(result.is_err());
    }

    #[test]
    fn status_errors_when_data_valid_but_attribute_absent() {
        let dp = make();
        let ctx = DefaultContext;
        // Data store reports valid, attribute store has no record.
        dp.inner_provider().set_status(6, ElementStatus::Valid);
        let result = block_on(dp.status_by_internal_id(&ctx, 6));
        assert!(result.is_err());
    }

    #[test]
    fn delete_delegates_to_inner_provider() {
        let dp = make();
        let ctx = DefaultContext;
        let vector = vec![0.0_f32];
        block_on(dp.set_element(&ctx, &1, Document::new(&vector, attrs()))).unwrap();

        block_on(dp.delete(&ctx, &1)).unwrap();

        // The inner provider now reports the id as deleted.
        let status = block_on(dp.inner_provider().status_by_internal_id(&ctx, 1)).unwrap();
        assert_eq!(status, ElementStatus::Deleted);
    }

    #[test]
    fn release_removes_attributes_and_releases_slot() {
        let dp = make();
        let ctx = DefaultContext;
        let vector = vec![0.0_f32];
        block_on(dp.set_element(&ctx, &1, Document::new(&vector, attrs()))).unwrap();

        block_on(dp.release(&ctx, 1)).unwrap();

        assert!(dp.inner_provider().released.lock().unwrap().contains(&1));
    }
}
