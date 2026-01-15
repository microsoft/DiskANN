# Inverted Index with Range Query Support - Architecture & Design

## Overview

This PR implements a **trait-based inverted index architecture** with support for **equality and range queries** on metadata fields. The design uses **order-preserving encoding** to enable efficient range queries (e.g., `age >= 25 AND age < 50`) while maintaining flexibility through composable abstractions.

## Architecture

The implementation consists of **five main abstraction layers** with clear separation between **write** and **read** operations:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Application Layer                                 │
│   Writes: insert/update/delete       |       Reads: query evaluation       │
└──────────────┬───────────────────────┴──────────────────┬──────────────────┘
               │                                           │
               │ WRITE PATH                                │ READ PATH
               │                                           │
               ▼                                           ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│   InvertedIndex Trait        │              │   QueryEvaluator Trait       │
│   (Write Operations)         │              │   (Query Operations)         │
│                              │              │                              │
│ • insert(doc_id, attrs)      │              │ • evaluate_query(expr)       │
│ • delete(doc_id)             │              │      → PostingList           │
│ • update(doc_id, attrs)      │              │ • is_match(doc_id, expr)     │
│ • batch_insert([...])        │              │      → bool                  │
│ • batch_delete([...])        │              │ • count_matches(expr)        │
│ • batch_update([...])        │              │      → usize                 │
│                              │              │                              │
│ Modifies index data          │              │ Evaluates AST expressions    │
└──────────────┬───────────────┘              └──────────────┬───────────────┘
               │                                             │
               │ directly uses                               │ uses
               │                                             │
               ▼                                             ▼
               │                              ┌──────────────────────────────┐
               │                              │ PostingListProvider Trait    │
               │                              │ (Read-Only Access)           │
               │                              │                              │
               │                              │ • get_posting_list(field,    │
               │                              │       value)                 │
               │                              │      → Option<PostingList>   │
               │                              │                              │
               │                              │ Fetches doc IDs (read-only)  │
               │                              └──────────────┬───────────────┘
               │                                             │
               │                                             │ uses
               │                                             │
               └─────────────────────┬───────────────────────┘
                                     │
                                     │ Both paths converge here
                                     │
            ┌────────────────────────┴────────────────────────┐
            │                                                  │
            ▼                                                  ▼
┌───────────────────────┐                        ┌────────────────────────┐
│   KeyCodec Trait      │                        │  PostingList Trait     │
│                       │                        │                        │
│ encode_field_value    │                        │ • empty() / insert()   │
│  (Order-Preserving)   │                        │ • remove() / contains()│
│                       │                        │ • union() / intersect()│
│ "age" + 25            │                        │ • difference()         │
│      ↓                │                        │ • len() / is_empty()   │
│ "age\0I8000...019"    │                        │ • serialize()          │
│                       │                        │ • deserialize()        │
└───────────┬───────────┘                        └────────────┬───────────┘
            │                                                  │
            │ Encodes keys                                     │ Stores doc IDs
            │                                                  │
            └──────────────────┬───────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │    KvStore Trait        │
                    │                         │
                    │  • get(key) → value     │
                    │  • set(key, value)      │
                    │  • del(key)             │
                    │  • range(start..end)    │
                    │  • batch_set([...])     │
                    │  • batch_del([...])     │
                    │                         │
                    │  key → value storage    │
                    │  "age\0I8..." → [blob]  │
                    └────────────┬────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │   Storage Backends (Pluggable)       │
              │                                      │
              │  • BfTree (persistent, on-disk)     │
              │  • BTreeMap (in-memory, for tests)  │
              │  • Future: RocksDB, LMDB, etc.      │
              └──────────────────────────────────────┘
```

**Key Data Flow**:

**Write Path** (InvertedIndex):
```
InvertedIndex.insert(doc_id=123, {"age": 25})
      │
      ├─> KeyCodec.encode("age", 25) → "age\0I8000000000000019" (key bytes)
      │
      ├─> PostingList{doc_ids}.serialize() → [...] (value bytes)
      │
      └─> KvStore.set("age\0I8000000000000019", [...])
              │
              └─> Storage Backend (BfTree/BTreeMap/etc.)
```

**Read Path** (PostingListProvider → QueryEvaluator):
```
QueryEvaluator.evaluate_query(age >= 25)
      │
      └─> PostingListProvider.get_posting_list("age", 25)
              │
              ├─> KeyCodec.encode("age", 25) → "age\0I8000000000000019"
              │
              ├─> KvStore.get("age\0I8000000000000019") → [...]
              │       │
              │       └─> Storage Backend (BfTree/BTreeMap/etc.)
              │
              └─> PostingList::deserialize([...]) → RoaringBitmap{42, 98, 123}
```

---

## Core Traits & Their Responsibilities

### 1. **InvertedIndex Trait** (`traits/inverted_index_trait.rs`) - WRITE OPERATIONS

The main interface for **modifying** the inverted index.

**Purpose**: Defines write operations to store, update, and delete document metadata using an inverted index structure. This trait focuses exclusively on data modification, keeping write operations separated from read operations.

**Key Methods**:
```rust
pub trait InvertedIndex {
    type Error: std::error::Error + Send + Sync + 'static;
    type DocId: Copy + Into<usize> + From<usize> + std::fmt::Debug;

    // CRUD Operations
    fn insert(&mut self, doc_id: Self::DocId, attributes: &Attributes) -> Result<()>;
    fn delete(&mut self, doc_id: Self::DocId) -> Result<()>;
    fn update(&mut self, doc_id: Self::DocId, attributes: &Attributes) -> Result<()>;
    
    // Batch Operations (more efficient than individual calls)
    fn batch_insert(&mut self, pairs: &[(Self::DocId, Attributes)]) -> Result<()>;
    fn batch_delete(&mut self, doc_ids: &[Self::DocId]) -> Result<()>;
    fn batch_update(&mut self, pairs: &[(Self::DocId, Attributes)]) -> Result<()>;
}
```

**Data Flow**:
```
Document: {"age": 25, "city": "Seattle"}
    ↓ flatten to (field, value) pairs
    ↓
[("age", 25), ("city", "Seattle")]
    ↓ encode each pair
    ↓
["age\0I8000000000000019", "city\0SSeattle"]
    ↓ store in KV store
    ↓
KV Store: "age\0I8000000000000019" → RoaringBitmap{doc_id}
```

**Design Rationale**:
- **Write-Only Interface**: Keeps modification operations separate from query operations
- **Mutable Operations**: Requires `&mut self` to signal state changes
- **Batch Support**: Efficient bulk operations for better performance

---

### 2. **PostingListProvider Trait** (`traits/posting_list_provider.rs`) - READ OPERATIONS

The interface for **querying** the inverted index without modification.

**Purpose**: Provides read-only access to posting lists, enabling query operations without the ability to modify the index. This trait is the foundation for all query evaluation.

**Key Methods**:
```rust
pub trait PostingListProvider {
    type Error: std::error::Error + Send + Sync + 'static;
    type PostingList: PostingList;
    type DocId: Copy + Into<usize> + From<usize> + std::fmt::Debug;

    // Read-only access to posting lists
    fn get_posting_list(
        &self,
        field: &str,
        value: &RealValue,
    ) -> Result<Option<Self::PostingList>>;
}
```

**Design Rationale**:
- **Read-Only Interface**: Takes `&self` (immutable reference), no modifications allowed
- **Interface Segregation**: Clients that only need to read don't depend on write methods
- **Flexibility**: Can be implemented independently or alongside `InvertedIndex`

**Typical Implementation Pattern**:
```rust
// An index implementation provides both write and read capabilities
// Write operations use KeyCodec, KvStore, and PostingList directly
impl InvertedIndex for MyIndex {
    fn insert(&mut self, doc_id: Self::DocId, attributes: &Attributes) -> Result<()> {
        // Directly uses KeyCodec to encode keys
        // Directly uses KvStore to store data
        // Directly uses PostingList to manage doc IDs
    }
}

// Read operations are exposed through PostingListProvider
// PostingListProvider also uses KeyCodec and KvStore internally
impl PostingListProvider for MyIndex {
    fn get_posting_list(&self, field: &str, value: &RealValue) -> Result<Option<Self::PostingList>> {
        // Uses KeyCodec to encode lookup key
        // Uses KvStore to fetch data
        // Returns PostingList
    }
}

// QueryEvaluator uses PostingListProvider for data access
impl QueryEvaluator for MyIndex { /* query operations */ }
```

---

### 3. **QueryEvaluator Trait** (`traits/query_evaluator.rs`) - QUERY EVALUATION

The interface for evaluating complex query expressions.

**Purpose**: Evaluates abstract syntax tree (AST) expressions and determines which documents match query criteria. Built on top of `PostingListProvider` for data access.

**Key Methods**:
```rust
pub trait QueryEvaluator {
    type Error: std::error::Error + Send + Sync + 'static;
    type PostingList: PostingList;
    type DocId: Copy + Into<usize> + From<usize> + std::fmt::Debug;
    
    // Required method - evaluates AST expression to PostingList
    fn evaluate_query(&self, query_expr: &ASTExpr) -> Result<Self::PostingList>;
    
    // Provided methods with default implementations
    fn is_match(&self, doc_id: Self::DocId, expr: &ASTExpr) -> Result<bool> {
        let bs = self.evaluate_query(expr)?;
        Ok(bs.contains(doc_id.into()))
    }
    
    fn count_matches(&self, expr: &ASTExpr) -> Result<usize> {
        let bs = self.evaluate_query(expr)?;
        Ok(bs.len())
    }
}
```

**Design Rationale**:
- **Independent from InvertedIndex**: Can be implemented separately from write operations
- **Composability**: Uses `PostingListProvider` to fetch data, then applies query logic
- **Flexibility**: Users can implement custom query evaluation strategies

**Typical Usage**:
```rust
// Implement QueryEvaluator using PostingListProvider for data access
impl QueryEvaluator for MyIndex {
    fn evaluate_query(&self, expr: &ASTExpr) -> Result<Self::PostingList> {
        match expr {
            ASTExpr::And(subs) => { /* intersect results */ },
            ASTExpr::Or(subs) => { /* union results */ },
            ASTExpr::Compare { field, op } => {
                // Uses PostingListProvider to fetch data
                self.get_posting_list(field, value)?
            },
        }
    }
}

// Use the trait methods
let matches = index.evaluate_query(&query)?;  // Get all matching docs
let is_match = index.is_match(doc_id, &query)?;  // Check specific doc
let count = index.count_matches(&query)?;  // Count matches
```

---

### 4. **KeyCodec Trait** (`codec/key_codec.rs`)

Encodes field-value pairs into **sortable byte keys** for efficient range queries.

**Purpose**: Transform `(field, value)` pairs into lexicographically sortable keys that enable range scans.

**Key Innovation**: Order-preserving encoding using industry techniques:

#### Encoding Strategy

```rust
pub trait KeyCodec {
    fn encode_field_value_key(&self, field: &str, value: &RealValue) -> Vec<u8>;
}
```

**Key Format**:
```
<field>\0<type_prefix><encoded_value>

Examples:
  "age\0I8000000000000019"  → age = 25 (integer)
  "age\0Ic039000000000000"  → age = 25.5 (float)
  "name\0SAlice"            → name = "Alice"
  "active\0B1"              → active = true
```

#### Order-Preserving Encoding

**Integers** (Sign-bit flip technique):
```rust
fn encode_i64(n: i64) -> u64 {
    (n as u64) ^ 0x8000000000000000  // XOR with sign bit
}

// Result: -100 < -1 < 0 < 1 < 100 (lexicographically sorted)
i64::MIN  →  0x0000000000000000
-1        →  0x7FFFFFFFFFFFFFFE
0         →  0x8000000000000000
1         →  0x8000000000000001
i64::MAX  →  0xFFFFFFFFFFFFFFFF
```

**Floats** (IEEE 754 sortable encoding):
```rust
fn encode_f64(f: f64) -> u64 {
    let bits = f.to_bits();
    if (bits >> 63) == 0 {
        bits ^ 0x8000000000000000  // Positive: flip sign bit
    } else {
        !bits  // Negative: flip all bits
    }
}

// Result: -∞ < -1.0 < 0.0 < 1.0 < +∞
```

**Why This Matters**: These encodings ensure that:
```
age = 10  →  "age\0I800000000000000a"
age = 25  →  "age\0I8000000000000019"
age = 50  →  "age\0I8000000000000032"

When sorted lexicographically: 10 < 25 < 50 ✓
This enables efficient range scans: age >= 25
```

---

### 5. **KvStore Trait** (`traits/kv_store_traits.rs`)

Provides a universal key-value storage interface with range scan support.

**Purpose**: Abstract over different storage backends (in-memory, on-disk, distributed).

**Key Methods**:
```rust
pub trait KvStore: Send + Sync {
    // Point operations
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn set(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn del(&self, key: &[u8]) -> Result<()>;
    
    // Range scan (critical for range queries!)
    fn range<R>(&self, range: R) -> Result<KvIterator<'_>>
    where R: Into<KeyRange>;
    
    // Batch operations
    fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> Result<()>;
    fn batch_del(&self, keys: &[&[u8]]) -> Result<()>;
}
```

**Range Support**:
```rust
// Supports standard Rust range syntax
store.range(b"age\0I8000000000000019"..b"age\0I8000000000000032")
//           └─── age >= 25 ────┘    └─── age < 50 ────┘

// Returns lazy iterator over (key, value) pairs in range
for item in store.range(start..end)? {
    let (key, value) = item?;
    // Process matching documents
}
```

**Backend Flexibility**:
```rust
// In-memory (testing)
let store: Box<dyn KvStore> = Box::new(BTreeMapStore::new());

// On-disk persistent (production)
let store: Box<dyn KvStore> = Box::new(BfTreeStore::with_config(config));

// Future: RocksDB, LMDB, Redis, etc.
```

---

### 6. **PostingList Trait** (`traits/posting_list_provider.rs`)

Manages sets of document IDs (posting lists) for efficient set operations.

**Purpose**: Store which documents match a specific field-value pair.

**Key Methods**:
```rust
pub trait PostingList: Clone + Sized + Send + Sync {
    fn empty() -> Self;
    fn insert(&mut self, id: usize) -> bool;
    fn remove(&mut self, id: usize) -> bool;
    fn contains(&self, id: usize) -> bool;
    
    // Set operations for query processing
    fn union(&self, other: &Self) -> Self;       // OR
    fn intersect(&self, other: &Self) -> Self;   // AND
    fn difference(&self, other: &Self) -> Self;  // NOT
    
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(bytes: &[u8]) -> Result<Self>;
}
```

**Default Implementation**: `RoaringPostingList` (compressed bitmaps)
```rust
// Efficient storage: 1M documents = ~125KB (compressed)
// Fast operations: union/intersect in microseconds
pub struct RoaringPostingList {
    pub rb: RoaringBitmap,
}
```

---

## How Operations Work End-to-End

### 1. **Insert Operation**

Adds a document's metadata to the inverted index.

```rust
// Input: Document with metadata
let doc_id = 123;
let attributes = {
    "age": 25,
    "city": "Seattle",
    "score": 89.5
};

index.insert(doc_id, &attributes)?;
```

**Step-by-Step Flow**:

```
1. Flatten attributes to (field, value) pairs
   → [("age", 25), ("city", "Seattle"), ("score", 89.5)]

2. For each (field, value):
   
   a) Encode using KeyCodec
      "age\0I8000000000000019"      ← age = 25
      "city\0SSeattle"               ← city = "Seattle"
      "score\0Ic059666666666666"     ← score = 89.5
   
   b) Get existing posting list from KV store
      key = "age\0I8000000000000019"
      existing = store.get(key)? → Some(RoaringBitmap{42, 98})
   
   c) Update posting list with new doc_id
      posting_list = RoaringBitmap{42, 98}
      posting_list.insert(123) → RoaringBitmap{42, 98, 123}
   
   d) Serialize and store back
      bytes = posting_list.serialize()
      store.set(key, bytes)?

3. Save reverse mapping for efficient deletion
   key = "@R:123"
   value = serialize([
       "age\0I8000000000000019",
       "city\0SSeattle",
       "score\0Fc059666666666666"
   ])
   store.set(key, value)?
```

**Result**: Document 123 is now indexed and searchable by all its fields.

---

### 2. **Delete Operation**

Removes a document from all posting lists.

```rust
index.delete(doc_id)?;
```

**Step-by-Step Flow**:

```
1. Load reverse mapping
   key = "@R:123"
   keys = store.get(key)? → [
       "age\0I8000000000000019",
       "city\0SSeattle",
       "score\0Fc059666666666666"
   ]

2. For each key:
   
   a) Get posting list
      key = "age\0I8000000000000019"
      posting_list = deserialize(store.get(key)?)
      → RoaringBitmap{42, 98, 123}
   
   b) Remove doc_id
      posting_list.remove(123) → RoaringBitmap{42, 98}
   
   c) Update or delete
      if posting_list.is_empty():
          store.del(key)?  ← No more docs with age=25
      else:
          store.set(key, posting_list.serialize())?

3. Delete reverse mapping
   store.del("@R:123")?
```

**Result**: Document 123 is completely removed from the index.

---

### 3. **Equality Query**

Find all documents where `field = value`.

```rust
// Query: city = "Seattle"
let query = json!({"city": {"$eq": "Seattle"}});
let ast = parse_query_filter(&query)?;
let results = index.evaluate_query(&ast)?;
```

**Step-by-Step Flow**:

```
1. Parse query to AST
   ASTExpr::Compare {
       field: "city",
       op: CompareOp::Eq("Seattle")
   }

2. Encode lookup key
   codec.encode_field_value_key("city", "Seattle")
   → "city\0SSeattle"

3. Direct lookup in KV store
   key = "city\0SSeattle"
   bytes = store.get(key)? → Some([...])

4. Deserialize posting list
   posting_list = RoaringPostingList::deserialize(bytes)?
   → RoaringBitmap{42, 98, 123, 456}

5. Return results
   → PostingList with doc_ids: [42, 98, 123, 456]
```

**Complexity**: O(1) - Single key lookup!

---

### 4. **Range Query** (The Key Innovation)

Find all documents where `field >= value1 AND field < value2`.

```rust
// Query: age >= 25 AND age < 50
let query = json!({
    "$and": [
        {"age": {"$gte": 25}},
        {"age": {"$lt": 50}}
    ]
});
let results = index.evaluate_query(&query)?;
```

**Step-by-Step Flow**:

```
1. Parse to AST
   ASTExpr::And([
       ASTExpr::Compare { field: "age", op: Gte(25.0) },
       ASTExpr::Compare { field: "age", op: Lt(50.0) }
   ])

2. Evaluate first condition: age >= 25
   
   a) Encode start key
      codec.encode_field_value_key("age", 25)
      → "age\0I8000000000000019"
   
   b) Create end key (upper bound for integers)
      → "age\0Iffffffffffffffffffffffff"
   
   c) Range scan over KV store
      for (key, value) in store.range(start_key..end_key)? {
          posting_list = PostingList::deserialize(value)?
          result = result.union(posting_list)
      }
      
      Scans over:
      "age\0I8000000000000019" → RoaringBitmap{42, 98}      (age=25)
      "age\0I800000000000001e" → RoaringBitmap{123}         (age=30)
      "age\0I8000000000000023" → RoaringBitmap{456, 789}    (age=35)
      "age\0I8000000000000028" → RoaringBitmap{234}         (age=40)
      "age\0I800000000000002d" → RoaringBitmap{567}         (age=45)
      ...
      
   d) Union all posting lists
      result = {42, 98} ∪ {123} ∪ {456, 789} ∪ {234} ∪ {567}
             = RoaringBitmap{42, 98, 123, 234, 456, 567, 789}

3. Evaluate second condition: age < 50
   
   Similar process but with different range:
   start_key = "age\0I00000000000000000" (minimum)
   end_key = "age\0I8000000000000032"    (age=50, exclusive)
   
   result2 = RoaringBitmap{42, 98, 123, 234, 456, 567, 789, ...}

4. Combine with AND
   final_result = result1.intersect(result2)
   → Documents with 25 <= age < 50

5. Return results
   → [42, 98, 123, 234, 456, 567, 789]
```

**Complexity**: O(log N + K) where N = total keys, K = keys in range

**Why This Works**:
- Order-preserving encoding ensures lexicographic sort = numeric sort
- Range scan in KV store is highly optimized (B-Tree seek)
- Set operations (union/intersect) are fast with RoaringBitmaps

---

### 5. **Complex Query with AND/OR**

Combining multiple conditions.

```rust
// Query: (age >= 25 AND age < 50) AND city = "Seattle"
let query = json!({
    "$and": [
        {"age": {"$gte": 25}},
        {"age": {"$lt": 50}},
        {"city": {"$eq": "Seattle"}}
    ]
});
```

**Query Execution Tree**:

```
                  AND (intersect)
                /      |       \
               /       |        \
        age >= 25   age < 50   city = "Seattle"
        (range)     (range)    (point lookup)
            |           |           |
            v           v           v
        {42,98,    {42,98,      {42,123,
         123,234,   123,234,     456,890}
         456,567}   456,567,
                    890}

Step 1: Evaluate age >= 25
  → posting_list1 = {42, 98, 123, 234, 456, 567, ...}

Step 2: Evaluate age < 50
  → posting_list2 = {42, 98, 123, 234, 456, 567, 890, ...}

Step 3: Evaluate city = "Seattle"
  → posting_list3 = {42, 123, 456, 890}

Step 4: Intersect all (AND operation)
  result = posting_list1 ∩ posting_list2 ∩ posting_list3
         = {42, 98, 123, 234, 456, 567} ∩ {42, 98, 123, 234, 456, 567, 890} ∩ {42, 123, 456, 890}
         = {42, 123, 456}

Final: Documents [42, 123, 456] match all conditions
```

**Optimization**: Early termination
```rust
let mut acc = self.evaluate_query(&conditions[0])?;
for condition in conditions.iter().skip(1) {
    let result = self.evaluate_query(condition)?;
    acc = acc.intersect(&result);
    
    if acc.is_empty() {
        break;  // Short-circuit: no results possible
    }
}
```

---

## Type Handling: Integer vs Float

**Challenge**: Queries use `f64` (e.g., `{"age": {"$gte": 25}}` → 25.0), but data might be stored as integers.

**Solution**: Dual-type scanning

```rust
fn range_query_gte(&self, field: &str, value: f64) -> Result<PostingList> {
    let mut result = PostingList::empty();
    
    // Case 1: If value is whole number, scan integer range
    if value.fract() == 0.0 {
        let int_value = value as i64;
        let int_key = encode("age", int_value);  // "age\0I8000000000000019"
        result = result.union(scan_range(int_key..int_end));
    }
    
    // Case 2: Always scan float range
    let float_key = encode("age", value);  // "age\0Fc039000000000000"
    result = result.union(scan_range(float_key..float_end));
    
    // Union ensures we find data regardless of storage type
    Ok(result)
}
```

**Example**:
```
Data inserted as: {"age": 25}  → stored as "age\0I8000000000000019"
Query: {"age": {"$gte": 25.0}} → scans both "age\0I..." and "age\0F..." ranges
Result: Finds the integer-encoded data ✓
```

---

## Pluggable Storage Backends

The `KvStore` trait enables swapping storage implementations without changing application code.

### Current Implementations

**1. BTreeMapStore (In-Memory)**
```rust
// For testing and development
let store = BTreeMapStore::new();
store.set(b"key", b"value")?;

// Native range support with BTreeMap
impl KvStore for BTreeMapStore {
    fn range<R>(&self, range: R) -> Result<KvIterator<'_>> {
        let data = self.data.read().unwrap();
        let items: Vec<_> = data
            .iter()
            .filter(|(k, _)| key_range.contains(k))
            .map(|(k, v)| Ok((k.clone(), v.clone())))
            .collect();
        Ok(Box::new(items.into_iter()))
    }
}
```

**2. BfTreeStore (On-Disk)**
```rust
// For persistent storage
let config = Config::new("./data", 1024 * 1024 * 32);
let store = BfTreeStore::with_config(config);

impl KvStore for BfTreeStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let tree = self.tree.read().unwrap();
        let mut buffer = vec![0u8; 4096];
        match tree.read(key, &mut buffer) {
            LeafReadResult::Found(n) => {
                buffer.truncate(n as usize);
                Ok(Some(buffer))
            }
            LeafReadResult::NotFound => Ok(None),
            LeafReadResult::InvalidKey => bail!("Invalid key"),
        }
    }
    
    fn range<R>(&self, range: R) -> Result<KvIterator<'_>> {
        // TODO: Use bf-tree's native scan() method
        // For now returns empty (implementation needed)
        Ok(Box::new(std::iter::empty()))
    }
}
```

### Adding New Backends

To add RocksDB support:

```rust
pub struct RocksDbStore {
    db: Arc<RocksDB>,
}

impl KvStore for RocksDbStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(self.db.get(key)?)
    }
    
    fn set(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put(key, value)?;
        Ok(())
    }
    
    fn del(&self, key: &[u8]) -> Result<()> {
        self.db.delete(key)?;
        Ok(())
    }
    
    fn range<R>(&self, range: R) -> Result<KvIterator<'_>> {
        let key_range = range.into();
        let iter = self.db.iterator(IteratorMode::From(
            &key_range.start,
            Direction::Forward
        ));
        
        let filtered = iter
            .take_while(|(k, _)| key_range.contains(k))
            .map(|(k, v)| Ok((k.to_vec(), v.to_vec())));
        
        Ok(Box::new(filtered))
    }
    
    fn batch_set(&self, entries: &[(&[u8], &[u8])]) -> Result<()> {
        let mut batch = WriteBatch::default();
        for (k, v) in entries {
            batch.put(k, v);
        }
        self.db.write(batch)?;
        Ok(())
    }
}
```

---

## Reference Implementation

The codebase includes a complete reference implementation in test-only code:

**Location**: `traits/inverted_index_trait.rs` → `mod reference_impl`

**Purpose**: Educational example showing how to implement all traits together.

```rust
#[cfg(test)]
mod reference_impl {
    pub struct GenericInvertedIndex<S: KvStore, PL: PostingList, K: KeyCodec> {
        store: Arc<S>,
        _pl: PhantomData<PL>,
        _kc: PhantomData<K>,
    }
    
    impl<S, PL, K> InvertedIndex for GenericInvertedIndex<S, PL, K> {
        // Full implementation with:
        // - Insert/delete/update
        // - Equality queries
        // - Range queries (gte, gt, lte, lt)
        // - Complex queries (AND/OR)
        // - Reverse mapping for efficient deletion
    }
    
    #[cfg(test)]
    mod tests {
        // 11 comprehensive tests covering:
        // - Basic CRUD operations
        // - Query evaluation
        // - Range queries
        // - Combined queries
    }
}
```
 

---

## Performance Characteristics

### Insert Operation
- **Complexity**: O(F × log N) where F = fields, N = total keys
- **Breakdown**:
  - Encoding: O(1) per field
  - KV store get/set: O(log N) per field (B-Tree)
  - Posting list update: O(1) amortized (RoaringBitmap)
  
### Delete Operation  
- **Complexity**: O(F × log N)
- **Optimization**: Reverse mapping avoids scanning all keys

### Equality Query
- **Complexity**: O(log N) - Single key lookup
- **No table scan needed!**

### Range Query
- **Complexity**: O(log N + K) where K = keys in range
  - B-Tree seek to start: O(log N)
  - Sequential scan: O(K)
  - No parsing or filtering needed (keys are pre-sorted)
  - Can use bucket range for optimization

### Complex Query (AND/OR)
- **Complexity**: O(M × (log N + K)) where M = conditions
- **Optimization**: Early termination, bitmap set operations

---

## Design Benefits

### 1. **Separation of Concerns**
- **InvertedIndex**: Write operations (insert/delete/update) - directly uses KeyCodec, KvStore, and PostingList
- **PostingListProvider**: Read operations (get_posting_list) - also uses KeyCodec and KvStore, but read-only
- **QueryEvaluator**: Query evaluation logic - uses PostingListProvider for data access
- **KeyCodec**: Encoding strategy - shared by both write and read paths
- **KvStore**: Storage backend - shared by both write and read paths
- **PostingList**: Document ID set operations - used by all layers

### 2. **Testability**
```rust
// Unit test with mock KV store
let store = MockKvStore::new();
let index = MyInvertedIndex::new(store);

// Integration test with real storage
let store = BfTreeStore::with_config(config);
let index = MyInvertedIndex::new(store);
```

### 3. **Flexibility**
- **Storage**: Swap BTreeMap ↔ BfTree ↔ RocksDB
- **Encoding**: Try different strategies without rewriting queries
- **Posting Lists**: Replace RoaringBitmap with custom implementation

### 4. **Performance**
- **Order-preserving encoding**: Range queries without table scans
- **Lazy iterators**: Process data incrementally
- **Backend optimization**: Each storage engine can use native features

### 5. **Extensibility**
```rust
// Add new query types without changing storage
impl InvertedIndex for MyIndex {
    fn evaluate_query(&self, expr: &ASTExpr) -> Result<PostingList> {
        match expr {
            ASTExpr::Compare { op: Near(distance), .. } => {
                // Geospatial query using same infrastructure
                self.range_query_near(field, distance)
            }
            // ... existing cases
        }
    }
}
```

---

## Testing

### Test Coverage

**65 tests total**, including:

1. **Key Codec Tests** (8 tests)
   - Order preservation for integers, floats, strings, booleans
   - Edge cases (min/max values, zero, negative numbers)
   - Type isolation (different types don't collide)

2. **KV Store Tests** (10 tests)
   - Basic CRUD operations
   - Range scans (inclusive, exclusive, unbounded)
   - Batch operations
   - Empty range handling
   - Idempotent deletes

3. **Inverted Index Tests** (11 tests in reference_impl)
   - Insert and retrieval
   - Update and delete
   - Bulk insert
   - Query evaluation (AND/OR)
   - Range queries (all 4 operators: <, <=, >, >=)
   - Combined queries
   - Reverse key serialization

4. **Parser Tests** (20+ tests)
   - Query parsing
   - AST construction
   - Error handling

5. **Utility Tests** (15+ tests)
   - JSON flattening
   - JSONL reading
   - Value conversion

### Running Tests

```bash
# All tests
cargo test --package label-filter --lib

# Specific modules
cargo test --package label-filter --lib -- codec::key_codec
cargo test --package label-filter --lib -- traits::kv_store_traits
cargo test --package label-filter --lib -- reference_impl

# With output
cargo test --package label-filter --lib -- --show-output
```



---

## References

- **Order-Preserving Encoding**: FoundationDB Tuple Layer, Google Bigtable
- **Inverted Index**: "Introduction to Information Retrieval" (Manning et al.)
- **RoaringBitmaps**: [arXiv:1603.06549](https://arxiv.org/abs/1603.06549)
- **IEEE 754 Sortable Floats**: PostgreSQL B-Tree implementation
- **KV Store Abstractions**: LevelDB, RocksDB APIs

