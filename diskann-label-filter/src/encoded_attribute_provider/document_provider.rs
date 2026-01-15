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
    DP: DataProvider + Delete + SetElement<VT>,
    AS: AttributeStore<DP::InternalId> + AsyncFriendly,
    VT: Sync + Send,
{
    type SetError = ANNError;
    type Guard = <DP as SetElement<VT>>::Guard;

    async fn set_element(
        &self,
        context: &Self::Context,
        id: &Self::ExternalId,
        element: &Document<'_, VT>,
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
