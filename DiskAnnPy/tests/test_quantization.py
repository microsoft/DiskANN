# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import unittest
from typing import Tuple

import numpy as np

from diskannpy import _diskannpy as dap


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_vectors(rows: int, dim: int, *, seed: int = 42) -> np.ndarray:
    return _rng(seed).standard_normal((rows, dim)).astype(np.float32)


def _assert_relative_error(
    testcase: unittest.TestCase,
    actual: float,
    approx: float,
    threshold: float,
) -> None:
    testcase.assertTrue(math.isfinite(approx), f"approx distance is not finite: {approx}")
    # Use absolute error normalized by max magnitude to handle signed values (e.g., inner product)
    magnitude = max(abs(actual), abs(approx), 1e-9)
    relative_error = abs(actual - approx) / magnitude
    testcase.assertLessEqual(
        relative_error,
        threshold,
        f"relative error {relative_error:.4f} exceeds threshold {threshold} "
        f"(actual={actual:.6f}, approx={approx:.6f})",
    )


def _compute_actual_distance(metric: str, lhs: np.ndarray, rhs: np.ndarray) -> float:
    if metric == "l2":
        diff = lhs - rhs
        return float(np.sum(diff * diff, dtype=np.float64))
    if metric == "inner_product":
        return -1.0 * float(np.dot(lhs, rhs))
    if metric == "cosine": 
        return 1.0 - float(np.dot(lhs, rhs) / (np.linalg.norm(lhs) * np.linalg.norm(rhs)))
    raise AssertionError(f"unexpected metric: {metric}")


def _exercise_quantizer(
    testcase: unittest.TestCase,
    quantizer,
    vectors: np.ndarray,
    preprocess_fn,
    distances_fn,
    *,
    threshold: float,
    metrics: Tuple[str, ...] = ("l2", "inner_product"),
    query_index: int = 1,
    target_index: int = 2,
) -> None:
    # compress_batch now returns a 2D uint8 numpy array (n, bytes_per_vector)
    compressed = quantizer.compress_batch(vectors)
    testcase.assertEqual(compressed.shape[0], vectors.shape[0])
    testcase.assertEqual(compressed.shape[1], quantizer.bytes_per_vector)
    testcase.assertEqual(compressed.dtype, np.uint8)

    # decompress expects a 1D array (single vector's bytes)
    roundtrip = np.array(quantizer.decompress(list(compressed[0])), dtype=np.float32)
    testcase.assertEqual(roundtrip.shape[0], quantizer.output_dim)
    testcase.assertTrue(np.all(np.isfinite(roundtrip)))

    lhs_source = vectors[target_index]
    rhs_source = vectors[query_index]

    for metric in metrics:
        preprocessed = preprocess_fn(metric, query_index)
        # distances_batch takes a 2D array and returns 1D array of distances
        distances = distances_fn(metric, preprocessed, compressed)
        testcase.assertEqual(distances.shape, (compressed.shape[0],))
        testcase.assertEqual(distances.dtype, np.float32)

        # Check distance for target_index using relative error
        approx = float(distances[target_index])
        actual = _compute_actual_distance(metric, rhs_source, lhs_source)
        _assert_relative_error(testcase, actual, approx, threshold)


def _threshold_for_bit_width(bit_width: int) -> float:
    """Return relative error threshold based on bit width."""
    if bit_width == 1:
        return 0.5  # 1-bit has very high quantization error
    if bit_width == 2:
        return 0.25
    if bit_width == 4:
        return 0.1
    return 0.05  # 8-bit


class TestMinMaxQuantizer(unittest.TestCase):
    def test_quantization_workflow_across_bit_widths(self) -> None:
        vectors = _make_vectors(16, 12, seed=7)

        for bit_width in (1, 2, 4, 8):
            with self.subTest(bit_width=bit_width):
                quantizer = dap.MinMaxQuantizer(bit_width, 1.0, vectors.shape[1])

                self.assertEqual(quantizer.dim, vectors.shape[1])
                self.assertEqual(quantizer.bit_width, bit_width)

                def preprocess(metric: str, query_idx: int, *, q=quantizer):
                    return q.preprocess(vectors[[query_idx]])

                def distances(metric: str, preprocessed, compressed, *, q=quantizer):
                    return q.distances_batch(preprocessed, compressed, metric)

                threshold = _threshold_for_bit_width(bit_width)
                _exercise_quantizer(
                    self, quantizer, vectors, preprocess, distances,
                    threshold=threshold, metrics=("l2",)
                )

    def test_double_hadamard_transform(self) -> None:
        dim = 11
        vectors = _make_vectors(24, dim, seed=19)
        quantizer = dap.MinMaxQuantizer(
            4,
            0.85,
            dim,
            transform="double_hadamard",
            target_behavior="natural",
            rng_seed=5,
        )

        self.assertGreaterEqual(quantizer.output_dim, dim)

        def preprocess(metric: str, query_idx: int, *, q=quantizer):
            return q.preprocess(vectors[[query_idx]])

        def distances(metric: str, preprocessed, compressed, *, q=quantizer):
            return q.distances_batch(preprocessed, compressed, metric)

        _exercise_quantizer(
            self, quantizer, vectors, preprocess, distances,
            threshold=0.1, metrics=("l2","cosine",  "inner_product")
        )


class TestProductQuantizer(unittest.TestCase):
    def test_training_compression_and_distances(self) -> None:
        dim = 10
        training = _make_vectors(256, dim, seed=13)  # More training data
        queries = _make_vectors(12, dim, seed=17)

        quantizer = dap.ProductQuantizer(training, num_chunks=5, num_centers=64, lloyds_iters=10, seed=11)

        self.assertEqual(quantizer.dim, dim)
        self.assertEqual(quantizer.output_dim, dim)
        self.assertEqual(quantizer.bit_width, 8)
        self.assertEqual(quantizer.num_chunks, 5)
        self.assertEqual(quantizer.num_centers, 64)
        self.assertEqual(quantizer.bytes_per_vector, 5)

        def preprocess(metric: str, query_idx: int, *, q=quantizer):
            return q.preprocess(queries[[query_idx]], metric)

        def distances(metric: str, preprocessed, compressed, *, q=quantizer):
            return q.distances_batch(preprocessed, compressed)

        # PQ has higher quantization error; only test L2 which is more stable
        _exercise_quantizer(
            self, quantizer, queries, preprocess, distances,
            threshold=0.5, metrics=("l2", "cosine", "inner_product")
        )
