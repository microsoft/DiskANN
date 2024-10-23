# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import unittest
import shutil
import tempfile

import numpy as np

from fixtures import random_vectors, vectors_as_temp_file

import diskannpy as dap


class TestVectorsFromFile(unittest.TestCase):
    def test_in_mem(self):
        expected = random_vectors(10_000, 100, dtype=np.float32)
        with vectors_as_temp_file(expected) as vecs_file:
            actual = dap.vectors_from_file(vecs_file, dtype=np.float32)
            self.assertTrue((expected == actual).all(), f"{expected == actual}\n{expected}\n{actual}")

    def test_memmap(self):
        expected = random_vectors(10_000, 100, dtype=np.float32)
        with vectors_as_temp_file(expected) as vecs_file:
            with tempfile.NamedTemporaryFile() as vecs_file_copy:
                shutil.copyfile(vecs_file, vecs_file_copy)
                actual = dap.vectors_from_file(
                    vecs_file,
                    dtype=np.float32,
                    use_memmap=True
                )
                self.assertTrue((expected == actual).all(), f"{expected == actual}\n{expected}\n{actual}")
                # windows refuses to allow 2 active handles via memmap to touch the same file
                # that's why we made a copy of the file itself and are using the copy here to test
                # the read+append(inmem) 
                actual = dap.vectors_from_file(
                    vecs_file_copy,
                    dtype=np.float32,
                    use_memmap=True,
                    mode="r+"
                )
                self.assertTrue((expected == actual).all(), f"{expected == actual}\n{expected}\n{actual}")


if __name__ == '__main__':
    unittest.main()
