import mmh3

class BloomFilter:
    def __init__(self, num_bits, num_hashes):
        self.size = num_bits
        self.hash_count = num_hashes
        self.bit_array = [0] * self.size

    def __len__(self):
        return self.size

    def to_str(self, delim=','):
        inds = [str(i) for i in range(len(self.bit_array)) if self.bit_array[i] == 1]
        return delim.join(inds)

    def add(self, item):
        for i in range(self.hash_count):
            digest = self.hash_i(str(item), i) % self.size
            self.bit_array[digest] = 1

    def check(self, item):
        for i in range(self.hash_count):
            digest = self.hash_i(item, i) % self.size
            if self.bit_array[digest] == 0:
                return False
        return True

    @staticmethod
    def hash_i(item, i):
        return mmh3.hash(item, i)
