# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import unittest

import numpy as np

from fixtures import random_vectors, vectors_as_temp_file

import diskannpy as dap


class TestVectorsFromFile(unittest.TestCase):
    def test_in_mem(self):
        expected = random_vectors(10_000, 100, dtype=np.float32)
        with vectors_as_temp_file(expected) as vecs_file:
            actual = dap.vectors_from_file(vecs_file, dtype=np.float32)
            self.assertTrue((expected == actual).all())

    def test_memmap(self):
        expected = random_vectors(10_000, 100, dtype=np.float32)
        with vectors_as_temp_file(expected) as vecs_file:
            actual = dap.vectors_from_file(
                vecs_file,
                dtype=np.float32,
                use_memmap=True
            )
            self.assertTrue(all(expected == actual))
            actual = dap.vectors_from_file(
                vecs_file,
                dtype=np.float32,
                use_memap=True,
                mode="r+"
            )
            self.assertTrue((expected == actual).all())


if __name__ == '__main__':
    unittest.main()
