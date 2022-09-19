import atexit
import os
import subprocess
import time
import unittest

import numpy as np
import requests

from tempfile import TemporaryDirectory

from disk_ann_util import build_ssd_index


class TestSSDRestApi(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if "DISKANN_BUILD_DIR" not in os.environ:
            raise Exception("We require the environment variable DISKANN_BUILD_DIR be set to the diskann build directory on disk")
        diskann_build_dir = os.environ["DISKANN_BUILD_DIR"]
        cls._build_dir = TemporaryDirectory()
        # using a fixed build dir for now
        _testing_build_dir = "/home/dax/testing/"

        rng = np.random.default_rng(12345)  # adjust seed for new random numbers
        cls._working_vectors = rng.random((1000, 100), dtype=float)
        build_ssd_index(
            diskann_build_dir,
            _testing_build_dir,
            #cls._build_dir.name,
            cls._working_vectors
        )
        # now we have a built index, we should run the rest server
        rest_port = rng.integers(10000, 10100)
        cls._rest_address = f"http://127.0.0.1:{rest_port}/"

        ssd_server_path = os.path.join(diskann_build_dir, "tests", "restapi", "ssd_server")

        args = [
            ssd_server_path,
            cls._rest_address,
            "float",
            _testing_build_dir,
            #cls._build_dir.name,
            "100",
            "1"
        ]

        command_run = " ".join(args)
        print(f"Executing REST server startup command: {command_run}")

        cls._rest_process = subprocess.Popen(args)
        time.sleep(10)

        cls._cleanup_lambda = lambda: cls._rest_process.kill()

        # logically this shouldn't be necessary, but an open port is worse than some random gibberish in the
        # system tmp dir
        atexit.register(cls._cleanup_lambda)

    @classmethod
    def tearDownClass(cls):
        cls._cleanup_lambda()

    def _is_ready(self):
        return self._rest_process.poll() is None  # None means the process has no return status code yet

    def test_responds(self):
        query = [0.0] * 100
        json_payload = {
            "Ls": 256,  # moar power rabbit
            "query_id": 1234,
            "query": query,
            "k": 10
        }
        try:
            response = requests.post(self._rest_address, json=json_payload)
            if response.status_code != 200:
                raise Exception(f"DOOM, DOOM UPON US ALL {response}")
        except Exception:
            raise Exception(f"Rest process status code is: {self._rest_process.poll()}")

