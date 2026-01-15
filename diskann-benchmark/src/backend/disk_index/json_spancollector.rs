/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::{Arc, Mutex};
use std::time::Duration;

use opentelemetry::trace::SpanId;
use opentelemetry::Value as OtelValue;

use opentelemetry_sdk::{
    error::OTelSdkResult,
    trace::{SpanData, SpanExporter},
};
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct JsonSpanCollector {
    spans: Arc<Mutex<Vec<SpanData>>>,
}

impl JsonSpanCollector {
    pub fn new() -> Self {
        Self {
            spans: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn to_hierarchical_json(&self) -> Value {
        let spans = self.spans.lock().unwrap();

        let mut children_map: HashMap<SpanId, Vec<&SpanData>> = HashMap::new();
        for span in spans.iter() {
            children_map
                .entry(span.parent_span_id)
                .or_default()
                .push(span);
        }

        let zero_id = SpanId::from_bytes([0u8; 8]);
        let root_spans = children_map.get(&zero_id).cloned().unwrap_or_default();

        let mut json_roots = Vec::new();
        for root_span in root_spans {
            json_roots.push(build_span_tree(root_span, &children_map));
        }

        json!({ "spans": json_roots })
    }
}

fn build_span_tree(span: &SpanData, children_map: &HashMap<SpanId, Vec<&SpanData>>) -> Value {
    let duration = span
        .end_time
        .duration_since(span.start_time)
        .unwrap_or(Duration::ZERO)
        .as_secs_f64();

    let mut metrics = serde_json::Map::new();
    metrics.insert("duration_seconds".to_string(), json!(duration));

    for attr in &span.attributes {
        metrics.insert(attr.key.to_string(), otel_value_to_json(&attr.value));
    }

    let mut span_obj = serde_json::Map::new();
    span_obj.insert("span_name".to_string(), json!(span.name));
    span_obj.insert("metrics".to_string(), json!(metrics));

    if let Some(children) = children_map.get(&span.span_context.span_id()) {
        if !children.is_empty() {
            let child_json: Vec<_> = children
                .iter()
                .map(|child| build_span_tree(child, children_map))
                .collect();
            span_obj.insert("children".to_string(), json!(child_json));
        }
    }

    Value::Object(span_obj)
}

impl SpanExporter for JsonSpanCollector {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.spans.lock().unwrap().extend(batch);
        Ok(())
    }

    fn shutdown_with_timeout(&mut self, _timeout: Duration) -> OTelSdkResult {
        Ok(())
    }
}

impl Default for JsonSpanCollector {
    fn default() -> Self {
        Self::new()
    }
}

fn otel_value_to_json(v: &OtelValue) -> Value {
    match v {
        OtelValue::Bool(b) => json!(*b),
        OtelValue::I64(i) => json!(*i),
        OtelValue::F64(f) => json!(*f),
        OtelValue::String(s) => json!(s.as_ref()),
        _ => json!(v.as_str()),
    }
}
