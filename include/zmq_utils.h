#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <zmq.h>

// Forward declarations for protobuf classes
namespace protoembedding
{
class NodeEmbeddingRequest;
class NodeEmbeddingResponse;
class NodeEmbedding;
} // namespace protoembedding

// ZMQ utility functions
bool fetch_embeddings_zmq(const std::vector<uint32_t> &node_ids, std::vector<std::vector<float>> &out_embeddings);

// Alias for backward compatibility
bool fetch_embeddings_http(const std::vector<uint32_t> &node_ids, std::vector<std::vector<float>> &out_embeddings);