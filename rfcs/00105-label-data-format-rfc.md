# RFC: Label Data Format for DiskANN

## 1. Requirements and Motivation

### 1.1 Current State and Problem

Currently, without a common format for label datasets in filtered vector search, there are already **three different variants** in the DiskANN codebase and many others in the broader research community. Each variant has its own specific scenarios and implementations, affecting label dataset reuse and filter search system/algorithm comparison. Without a standardized format for creating and sharing filtered vector search datasets with labels, researchers and engineers miss out on the consistency a common format would provide.

### 1.2. Format Requirements

- Enable extensibility to represent diverse metadata patterns. Allow flexibility for:
  - Single or multiple labels per item
  - Multiple data types (string, number, boolean, array, object)
  - Nested metadata structures
  - Table-style (SQL-like) and document-style (NoSQL) schemas
- Support comprehensive filtering syntax on all metadata patterns present in base labels.
- Support a unified format for both base dataset labels and query expressions that is easy to understand across the community.
- Allow easy extensibility for new query operators and new data types that are compatible with the current standard.

## 2. Format Overview

This RFC defines a **unified JSON-based format** for representing:
- **Vector Metadata Format**: Metadata associated with each vector in the base dataset.
- **Query Expression Format**: Search/filter conditions for each query vector.
---

### 2.1. Vector Metadata Format

#### Vector Metadata Format (EBNF)

```
BaseLabelFile   = LabelObject (\n LabelObject)*
LabelObject     = { "id": IdValue, Field* }
IdValue         = number
Field           = <string>: Value
Value           = string | number | boolean | null | [ Value, ... ] | { <string>: Value, ... } 
```

- Each line in the file is a single JSON object (label object).
- Each label object must have an `id` field (string or number).
- The file is not a valid JSON array, but a sequence of JSON objects separated by newlines (JSONL).

---

### Example (JSONL)

single label metadata JSON object
```
  {
    "id": 0,
    "category": "laptop",
    "brand": "Apple",
    "price": 1299.99,
    "quantity": 5,
    "in_stock": true,
    "rating": 4.8,
    "specifications": {
      "processor": "Intel i7",
      "ram": "16GB",
      "cores": 8,
      "base_clock": 2.6
    },
    "tags": ["premium", "gaming"]
  }
```

Files with multiple lines of JSON
```
{"id": 0, "category": "laptop", "brand": "Apple", "price": 1299.99, "quantity": 5, "in_stock": true, "rating": 4.8, "specifications": {"processor": "Intel i7", "ram": "16GB", "cores": 8, "base_clock": 2.6}, "tags": ["premium", "gaming"]}
{"id": 1, "category": "desktop", "brand": "Dell", "price": 899.99, "quantity": 12, "in_stock": true, "rating": 4.2, "specifications": {"processor": "Intel i5", "ram": "8GB", "cores": 4, "base_clock": 3.2}, "tags": ["budget", "office"]}
```

#### Notes
- **ID field**: Each label object should include an `id` field that uniquely identifies the corresponding vector in the base dataset.
- Nested objects are supported for hierarchical metadata.
- All standard JSON types are allowed (string, number, boolean, null, object, array).


---

## 3. Query Expression Format 

### Query Expression Format (EBNF)

```
QueryFile       = QueryObject (\n QueryObject)*
QueryObject     = { "query_id": IdValue,"filter": FilterExprLevel1 }
IdValue         =  number

FilterExprLevel1 = { (FieldName | LogicalOp): QueryValueLevel1, ... }
QueryValueLevel1 = Value | ComparisonOp | [ FilterExprLevel2, ... ]

FilterExprLevel2 = { (FieldName | LogicalOp): QueryValueLevel2, ... }
QueryValueLevel2 = Value | ComparisonOp

ComparisonOp    = { "$eq": Value } | { "$ne": Value } | { "$lt": Value } | { "$lte": Value } | { "$gt": Value } | { "$gte": Value } | { "$in": [Value, ...] }
Value           = string | number | boolean | null | [ Value, ... ] | { <string>: Value, ... } 
```

- Each line in the file is a single JSON object (query object).
- Each query object must have a `filter` field containing the query expression.
- The file is not a valid JSON array, but a sequence of JSON objects separated by newlines (JSONL).

---

### Example  
Single Query Expression 
```
  {
    "query_id": 0,
    "filter": {
      "category": {"$eq": "laptop"},
      "price": {"$lt": 1500.0},
      "quantity": {"$gte": 1},
      "in_stock": {"$eq": true},
      "rating": {"$gt": 4.5},
      "specifications.processor": {"$eq": "Intel i7"},
      "specifications.cores": {"$gte": 6},
      "$and": [
        {"brand": {"$eq": "Apple"}},
        {"price": {"$gte": 1000.0}}
      ],
      "$or": [
        {"category": {"$in": ["laptop", "desktop"]}},
        {"rating": {"$gt": 4.5}}
      ]
    }
  }
```
Query files with multiple queries 
```
{"query_id": 0, "filter": {"category": {"$eq": "laptop"}, "price": {"$lt": 1500.0}, "quantity": {"$gte": 1}, "in_stock": {"$eq": true}, "rating": {"$gt": 4.5}, "specifications.processor": {"$eq": "Intel i7"}, "specifications.cores": {"$gte": 6}, "$and": [{"brand": {"$eq": "Apple"}}, {"price": {"$gte": 1000.0}}], "$or": [{"category": {"$in": ["laptop", "desktop"]}}, {"rating": {"$gt": 4.5}}]}}
{"query_id": 1, "filter": {"category": {"$eq": "desktop"}, "price": {"$lt": 1000}, "specifications.cores": {"$eq": 4}, "tags": {"$in": ["budget", "office"]}}}
```

#### Supported Operators

- **Comparison**: `$eq`, `$ne`, `$lt`, `$lte`, `$gt`, `$gte`
- **Logical**: `$and`, `$or`, `$not`
- **Dot notation**: For nested fields (e.g., `specifications.processor`)

#### Notes
- Filters can be arbitrarily nested using logical operators.
- All filter keys and operators follow MongoDB-style syntax.


---

### 3.1. Ground Truth Format for Filtered Datasets

For filtered search, the ground truth results may not always contain exactly `k` matches due to filter constraints. The ground truth format must support variable result counts per query.

- First line is metadata of the groundtruth `distance` and `query_num`
- Following lines are groundtruth for each query 
  - The `count` field explicitly states how many IDs are present for each query.
  - The `ids` field contains the `id` values from the corresponding base label objects that match the query filter.
  - The `distances` field contains the distance value for the corresponding `id`.
  - The order of the array corresponds to the order of queries in the query label file.
### Example (JSONL)

```
{"distance_func":"l2", "query_num":2}
{"query_id": 0, "count": 2, "ids": [0,1], "distances": [0.234,0.235]}
{"query_id": 1, "count": 1, "ids": [0], "distances": [0.222]}
```
---

