# diskann-garnet Data Design

This document covers how index terms are stored and accessed in diskann-garnet. It assumes some prior knowledge of the DataProvider API in DiskANN which is the interface through which the core DiskANN algorithms access data.

## Garnet Storage

Garnet is essentially a fancy hashmap where values are stored under keys. Normally one interacts with Garnet through the Redis protocol, but diskann-garnet is directly linked in Garnet and given several methods it can use to read and write values. Detailed documentation of these is available in the [Garnet docs](https://github.com/microsoft/garnet/blob/vectorApiPoC/website/docs/dev/vector-sets.md#diskann-integration).

The available methods are read, write, delete, and read-modify-write (rmw).

### Callbacks

#### Read

The read method can be used to access a single value or multiple values. It takes a set of keys to read, and invokes a callback that gets access to a `&[u8]` of each value. Note that the read method does not return data directly; it simply invokes the supplied callback. The callback may access data in place or copy it out as desired. If a key does not exist, the callback will not be invoked for that key.

This design means that it is quite efficient as it can read or copy both full and partial values and when multiple keys are read, Garnet does prefetching to reduce memory latency of access.

#### Write

The write method always writes to a single key and must write the entire value.

#### Delete

The delete method deletes a single key/value pair from Garnet.

#### Read-Modify-Write (RMW)

The read-modify-write (rmw) method accesses a single key with a callback but allows the callback to modify the data in place. This operation happens under a lock so it is important that the callback be fast.

### Keys & Contexts

Garnet keys are arbitrary byte strings (e.g. `&[u8]`). The methods described above can use whatever keys they like to read and write data, however those methods also take a semi-opaque context which gives the operation a scope. For the most part this context is used for internal Garnet bookkeeping and is opaque, but the least significant 3 bits are available for diskann-garnet to use for its own scoping.

Diskann-garnet uses these bits to distinguish between differnet kinds of index data so that the same key can be used to fetch different kinds of data. For example, vector data might be stored under the same key, the vector ID, as neighbor lists by setting different context bits for the operation.

### Key Data Prefixing

In order to reduce allocations in the data access path in Garnet, Garnet needs some place to scribble state into during operations. It uses a single byte immediately preceding the first key byte for this purpose. This means that any key pointer given to Garnet access methods must contain valid space preceding the real key. For this reason, key data pointers are `* mut` and not `* const` and care must be taken to ensure the memory preceding that pointer is valid.

## Term Types

The data maintained by the index are referred to as terms. Full precision vectors are one kind of term,and neighbor lists are another kind of term.

The term types in diskann-garnet are: full precision vectors, neighbor lists, quantized vectors, attributes, metadata, and the internal and external ID mappings.

### Full Precision Vectors

*Key*: Internal ID as bytes

Full precision vectors are always a fixed size of `dimension * mem::size_of::<ElementType>()`. They are read or written whole or deleted.

In a quantized index, these vectors are used mainly during reranking in most configurations. In a full precision only index, they are the most accessed term.

### Neighbor Lists

*Key*: Internal ID as bytes

Neighbor lists are stored as a fixed size of `(max_neighbors + 1) * mem::size_of::<u32>()` where `max_neighbors` accounts for the graph slack factor. The final entry is the true length of the neighbor list.

For example, in a graph where the degree is 16 and the graph slack factor is 1.3, the size of a neighbor list would be `((16 * 1.3) as u32 + 1) * 4` bytes long. Using fixed size lists this way means that all neighbor list allocations are the same size.

In a typical index, these are accessed second most often after quantized vectors (in a quantized index) or full precision vectors (in a full precision only index).

### Quantized Vectors

*Key*: Internal ID as bytes

Quantized vectors a similar to full precision vectors in that they are fixed size and read/written as a whole, although they will often have a more complex representation that just an array of quantized elements.

In a quantized index, these vectors are the most often accessed piece of data and should be read in a batch when possible.

### Attributes

*Key*: Internal ID as bytes

When vectors are inserted by the Redis Vector Set API, an arbitrary JSON blob of attributes can be attached. These attributes are stored as a utf-8 string and read/written as a whole unit.

Attributes are ignored during normal searches but will be accessed during a filtered search. The Vector Set API allows users to request the attributes for search results, so even in a normal search these terms may be accessed in order to return the attributes in the results.

### Attributes Index

*Key*: Attribute name + Attribute value as bytes

When vectors are inserted by the Redis Vector Set API, and arbitrary JSON blob of attributes can be attached.

Attributes index should be created for fields with atrribute filtering need, a list of Internal ID are saved under the Attributes Index Key, if the attribute are present in the associated JSON for the Internal ID 


Garnet API --> Tsavorite --> VectorSet [DiskANN]--> Tsavorite RAWSTRING [today]


Garnet API --> Tsavorite --> VectorSet [DiskANN] --> Tsavorite RAWSTRING + Tsavorite RANGEINDEX [next-gen]


### Internal Terms

The are several terms in the index used for internal state management of the diskann-garnet provider.

#### Start Points

The start points are the same as other vectors, but the internal IDs of start points begin at `u32::MAX` and go downward. Currently a single start point is supported and will be a frozen copy of the first vector added to the index.

#### Metadata

Metadata is currently used for the free space map which manages used and available internal IDs.

##### Free Space Map

*Key*: b'_fsm' concatenated with FSM block number as bytes

The free space map is used to keep track of which internal IDs are allocated and in use. Please see the [ID Mapping](#id-mapping) section for more details on why mapped IDs are used.

The free space map is a series of blocks where each block is a string of 2-bit values representing the state of the corresponding internal ID. Free IDs are represented by `0b00`, used IDs by `0b01`, and deleted IDs by `0b10`. Blocks are created on demand during insert when they are needed.

During startup, the index will scan FSM blocks in sequence to restore state. It will update the correct bits in a FSM block whenever the state of an internal ID changes.

#### Internal ID Mapping

*Key*: Internal ID as bytes

Each internal ID corresponds to an external ID, which is a byte string of arbitrary length. The external IDs are stored unmodified and read/written as a whole.

Lookup of an external ID will happen during post processing when we return results to Garnet.

#### External ID Mapping

*Key*: External ID bytes

Each external ID corresponds to an internal ID which is a u32. The internal IDs are stored as 4 bytes and read/written as a whole.

Lookup of an internal ID will happen for things such as delete.

## ID Mapping

Garnet vector set IDs are arbitrary length byte strings natively. These are quite inefficient for indexing so we map each external ID to a `u32` internal ID. This imposes a maximum on the number of vectors that are indexable of `u32::MAX - 1` (the start point always consumes an ID).

When the DiskANN algorithm performs searches and other operations it works with internal IDs only. The external IDs are only used when returning data to the user (which has no concept of the internal IDs) or when asked to perform operations like delete on specific vectors which will be identified by their external ID. In order to convert back and forth for these occasions, lookup tables must be kept for the mapping.

In addition to lookup tables for mapping, diskann-garnet must also track the status of internal IDs so that IDs that are no longer in use may be reused by future vectors. The [free space map](#free-space-map) is used to track status for each internal ID, and the data structure that handles the FSM also maintains a short list of available IDs for re-use so that requests to Garnet are amortized.

## Concurrency Issues

Index operations are not atomic, but individual term accesses are. There is one compound atomic operation which Garnet provides which is read-modify-write. Due to the lack of atomic operations, several concurrency related issues can arise.

### Neighbor List Updates

The `append_vector()` function in the Data Provider API adds new neighbors to an existing vector's neighbor list. While most data providers do a read operation followed by a write of whole new neighbor list, Garnet's read-modify-write operation allows for atomics updates. Unlike many other data providers there is no chance that simultaneous updates to a neighbor list will cause neighbors to be discarded.

### Vector Terms vs. Other Terms

Vector terms are a bit special as failure to read a vector term is an expected operation due to the index not having atomic high level operations. During a search, it is quite possible for a vector close to the query to be found which is then deleted during the rest of the search operation. This can also happen when vectors are deleted as deletes do not fully expunge the vector's ID in every neighbor list in the index.

In cases where vectors are not found, transient errors are returned which can be ignored.

### Final Results

After a search operation a set of candidates is produced to return to the user along with their corresponding distances and optionally their attributes. It is possible that after the candidates are collected, those vectors may be mutated or deleted before they are returned to the user.

As DiskANN is an approximate algorithm, it is acceptable for some vectors which have been mutated to appear in the results. This will penalize recall slightly but is not technically wrong. In filtered search, however, mutation is more serious as returning vectors that do not match the filter criteria is incorrect. It is desireable to reduce the former and eliminate the latter problem as much as possible.

#### Deleted Vectors

Vectors which are deleted after candidates are found but before they are returned can be removed during post processing. Due to how in-place deletes work in DiskANN this post processing is normal and has existing post processors to handle it. However, it is always possible for vectors to be deleted after results are returned, before the user attempts to retrieve the corresponding document.

#### Mutated Vectors

Vectors which are deleted and replaced may cause both their vector data to change as well as their attributes. This can happen via concurrent operations on the index; for example, inserting a new vector with the same external ID will overwrite the existing vector. This can also occur as a consequence of in-place or consolidated deletes.

During deletes, the deleted vector's terms are removed, but they will be present in other vectors neighbor lists for some time. In the case of consolidated delete, these vector IDs will remain until consolidation is invoked, and for in-place deletes there is no specific event which enforces removal of the vector's ID from the graph. When search or other operations find the vector ID in a neighbor list and attempt to load it they may find either the vector is missing or that a different vector has now been inserted there. This should be ok as distances will be calculated on the new data and if it is far from the query it will be discarded.

Filtered search introduces two problems regarding attributes. The first is that the vector IDs and attributes for the vector need to match when results are returned the user. The second is that vector data and vectors attributes must match during the filtered search operation. Currently no guarantees are made about this as each term is stored separately.

It is possible to co-locate the appropriate vector term and the attribute term as a single value which may alleviate much of this problem. Because we can read partial values in the read callback, we can access the correct portion of the data (vector data would be stored first as it has a known, fixed size). However, we can alleviate only one of the two problems in an index with both full precision and quantized vector terms as we must choose whether the search operation should not see skew or whether the final reranking step should not see skew.


 