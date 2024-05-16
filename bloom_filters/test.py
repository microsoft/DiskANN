import math

hash_count = lambda m, n: (m / n) * math.log(2)
fp_rate = lambda k: (1/2) ** k

print(f"hc: {hash_count(256, 40)}")
print(f"fp: {fp_rate(hash_count(256, 40))}")
