import struct
import parse_common

def process_sector(file, data_type_code, nodes_per_sector, num_dims, running_node_id, num_nodes):
    sector_offset = 0
    num_nodes_read = 0
    nodes = []
    while num_nodes_read < nodes_per_sector and running_node_id < num_nodes:
        node = parse_common.Node(running_node_id, data_type_code, num_dims)
        node.load_from(file)

        num_nodes_read += 1
        nodes.append(node)
        running_node_id += 1
    return nodes


def parse_index(index_path_prefix, data_type_code, data_type_size, out_graph_csv):
    return parse_index_with_PQ_vectors(index_path_prefix, data_type_code, data_type_size, out_graph_csv, None)
        
def parse_index_with_PQ_vectors(index_path_prefix, data_type_code, data_type_size, out_graph_csv, compressed_vectors = None):
    disk_index_file_name = index_path_prefix + "_disk.index"

    with open(out_graph_csv, "w") as out_file:
        out_file.write("Id\tvector\tneighbours\n")

        with open(disk_index_file_name, "rb") as index_file:
            num_entries = struct.unpack('I', index_file.read(4))[0]
            num_dims = struct.unpack('I', index_file.read(4))[0]

            if num_dims != 1 or num_entries != 9:
                raise Exception("Mismatch in metadata. Expected 1 dimension and 9 entries. Got " + str(num_dims) + " dimensions and " + str(num_entries) + " entries.")

            # Read the metadata
            num_nodes = struct.unpack('Q', index_file.read(8))[0]
            num_dims = struct.unpack('Q', index_file.read(8))[0]
            medoid_id = struct.unpack('Q', index_file.read(8))[0]
            max_node_len = struct.unpack('Q', index_file.read(8))[0]
            nnodes_per_sector = struct.unpack('Q', index_file.read(8))[0]

            print("Index properties: " + str(num_nodes) + " nodes, " 
                  + str(num_dims) + " dimensions, medoid id: " 
                  + str(medoid_id) + ", max node length: " + str(max_node_len) 
                  + ", nodes per sector: " + str(nnodes_per_sector))
            

            #skip the first sector 
            index_file.seek(parse_common.SECTOR_LEN, 0)
            
            nodes_read = 0
            sector_num = 1
            while nodes_read < num_nodes:
                nodes = process_sector(index_file, data_type_code, nnodes_per_sector, num_dims, nodes_read, num_nodes)
                assert len(nodes) <= nnodes_per_sector
                for node in nodes:
                    if compressed_vectors is not None:
                        compressed_vector = compressed_vectors[node.id]
                        node.vector = compressed_vector
                    out_file.write(str(node))
                    out_file.write("\n")
                nodes_read += len(nodes)
                sector_num += 1
                index_file.seek(sector_num * parse_common.SECTOR_LEN, 0)
                if sector_num % 100 == 0:
                    print("Processed " + str(sector_num) + " sectors and " + str(nodes_read) + " nodes.")
    
    print("Processed " + str(nodes_read) + " points from index in " 
          + disk_index_file_name + " and wrote output to " + out_graph_csv)
