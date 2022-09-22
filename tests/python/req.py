import numpy as np
import requests

rng = np.random.default_rng(12345)

query = rng.random((100), dtype=float).tolist()
print(query)

host = "http://127.0.0.1:10067/"

json_payload = {
    "Ls": 256,  # moar power rabbit
    "query_id": 1234,
    "query": query,
    "k": 10
}

response = requests.post(host, json=json_payload)
if response.status_code != 200:
    raise Exception(f"DOOM, DOOM UPON US ALL {response}")

