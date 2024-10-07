import sys
import numpy as np
import queue

class DistanceStats:
    def __init__(self):
        self.close = []
        self.far = []
        self.intermediate = []
        self.uncategorized = []
        self.graphlevel = 0

    def categorize(self, min, max):
        one_third = (max - min )/2.0
        two_third = 2 * one_third
        for edge in self.uncategorized:
            if edge[1] < one_third:
                self.close.append(edge)
            elif edge[1] < two_third:
                self.intermediate.append(edge)
            else:
                self.far.append(edge)
        self.uncategorized = []

   
data = {} #id, vector map
graph = {} #id, [id1, id2, ...] map

#a and b are tuples of the form (vector, norm)
def cosine(a, b):
    return np.dot(a[0], b[0]) / (a[1]) * (b[1])

def l2_distance(a,b):
    return np.linalg.norm(a[0] - b[0])


def compute_distances(data, graph, start_node, dist_metric):
    navigation_queue = queue.Queue()
    navigation_queue.put((start_node, 0))
    classified_edges = {} 
    processed_nodes = set()

    max = sys.float_info.min
    min = sys.float_info.max
    while not navigation_queue.empty():
        current_node_and_level = navigation_queue.get()
        current_node = current_node_and_level[0]
        current_level = current_node_and_level[1]

        if current_node in processed_nodes: 
            continue
        
        classified_edges[current_node] = DistanceStats()
        classified_edges[current_node].graphlevel = current_level

        for edge_node in graph[current_node]:
            dist = dist_metric(data[current_node], data[edge_node])
            classified_edges[current_node].uncategorized.append((edge_node, dist))
            if dist > max:
                max = dist
            if dist < min:
                min = dist  

            #Because we know the graph has cycles, it is possible that we have already processed the edge node
            if edge_node not in processed_nodes:
                navigation_queue.put((edge_node, current_level + 1))
        
        processed_nodes.add(current_node)
    
    return classified_edges, max, min


def main(input_file, start_node, dist_metric_str, out_stats_prefix): 
    #read tsv file with JSON data
    firstline = True
    with open(input_file, 'r') as f:
        for line in f:
            if firstline:
                firstline = False
                continue
            line = line.strip().split('\t')
            id = int(line[0].strip())
            vector = np.array([float(x) for x in line[1].strip('[').strip(']').split(',')])
            data[id] = (vector, np.linalg.norm(vector))
            graph[id] = np.array([int(x) for x in line[2].strip('[').strip(']').split(',')])
    f.close()

    assert(len(data) == len(graph))
    print("Read {} rows from {}. Data dim: {}".format(len(data), input_file, len(data[0][0])))

    dist_metric = None
    if dist_metric_str == 'l2':
        dist_metric = l2_distance
    elif dist_metric_str == 'cosine':
        dist_metric = cosine
    else:
        raise ValueError("Invalid distance metric. Use l2 or cosine")
    
    classified_edges, max, min = compute_distances(data, graph, start_node, dist_metric)
    print("Max distance: {}, Min distance: {}".format(max, min))

    for node in classified_edges:
        classified_edges[node].categorize(min, max)

    out_stats_file = out_stats_prefix + '_stats.tsv'
    with open(out_stats_file, 'w') as f:
        for node in classified_edges:
            f.write(str(node) + '\t' + str(classified_edges[node].graphlevel) 
                        + '\t' + str(len(classified_edges[node].close)) 
                        + '\t' + str(len(classified_edges[node].intermediate)) 
                        + '\t' + str(len(classified_edges[node].far)) + '\n')
    f.close()

    # out_edges_file = out_stats_prefix + '_edges.tsv'
    # with open(out_edges_file, 'w') as f:
    #     for node in classified_edges:
    #         for edge in classified_edges[node].close:
    #             f.write(str(node) + '\t' + str(edge[0]) + '\t' + str(edge[1]) + '\t' + 'close' + '\n')
    #         for edge in classified_edges[node].intermediate:
    #             f.write(str(node) + '\t' + str(edge[0]) + '\t' + str(edge[1]) + '\t' + 'intermediate' + '\n')
    #         for edge in classified_edges[node].far:
    #             f.write(str(node) + '\t' + str(edge[0]) + '\t' + str(edge[1]) + '\t' + 'far' + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python EdgeClassifier.py <input_file> <start_node> <dist_metric (l2|cosine)`> <out_stats_prefix>")
        sys.exit(1)

    if sys.argv[3] != 'l2' and sys.argv[3] != 'cosine':
        print("Invalid distance metric. Use l2 or cosine")
        sys.exit(1)
    

    main(sys.argv[1], int(sys.argv[2].strip()), sys.argv[3], sys.argv[4])