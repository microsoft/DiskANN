import array
import struct

"""Constants"""
SECTOR_LEN = 4096
NUM_PQ_CENTROIDS = 256

class Node:
    def __init__(self, id, data_format_specifier, num_dims):
        self.id = id
        self.vector = None
        self.data_type = data_format_specifier
        self.num_dims = num_dims
        self.neighbors = None
    
    def __str__(self):
        if self.vector is None:
            raise Exception("Vector is not initialized")
        else:
            return str(self.id) + "\t" + str(self.vector.tolist()) + "\t" + str(self.neighbors.tolist()) 
    
    def load_from(self, file):
        self.vector = array.array(self.data_type)
        self.vector.fromfile(file, self.num_dims)
        num_neighbors = struct.unpack('I', file.read(4))[0]
        self.neighbors = array.array('I') #unsigned int neighbor ids. 
        self.neighbors.fromfile(file, num_neighbors)

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def add_vector_dim(self, vector_dim):
        self.vector.append(vector_dim)


class DataMat: 
    def __init__(self, array_format_specifier, datatype_size):
        self.num_rows = 0
        self.num_cols = 0
        self.data = None
        self.data_format_specifier = array_format_specifier
        self.data_type_size = datatype_size

    
    def load_bin(self, file_name):
        with open(file_name, "rb") as file:
            self.load_bin_from_opened_file(file)

    def load_bin_metadata_from_opened_file(self, file):
        self.num_rows = struct.unpack('I', file.read(4))[0]
        self.num_cols = struct.unpack('I', file.read(4))[0]
        print(file.name + ": #rows: " + str(self.num_rows) + ", #cols: " + str(self.num_cols))

    def load_bin_from_opened_file(self, file, file_offset_data=0): 
        file.seek(file_offset_data, 0)
        self.load_bin_metadata_from_opened_file(file)
        self.data = array.array(self.data_format_specifier)
        self.data.fromfile(file, self.num_rows*self.num_cols)

    def load_data_only_from_opened_file(self, file, num_rows, num_cols, file_offset_data=0):
        file.seek(file_offset_data, 0)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.data = array.array(self.data_format_specifier)
        self.data.fromfile(file, self.num_rows*self.num_cols)

    def __len__(self):
        return self.num_rows
    
 
    def __getitem__(self, key):
        return self.data[key*self.num_cols:(key+1)*self.num_cols]   


