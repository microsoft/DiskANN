import pickle
import numpy as np
import struct

def read_bin_file(bin_file, data_type=np.float32):
    """
    Read and validate the binary file containing vectors.
    """
    with open(bin_file, 'rb') as f:
        # Read the header (number of vectors and dimension)
        header = f.read(8)
        num_vectors, vector_size = struct.unpack('ii', header)
        print(f"Number of vectors: {num_vectors}")
        print(f"Vector size: {vector_size}")

        # Read the vector data
        data = np.fromfile(f, dtype=data_type)
        # data = np.fromfile(f, dtype=np.float32)
        print(f"Total floats read: {len(data)}")
        
        # Validate the data size
        expected_size = num_vectors * vector_size
        if len(data) != expected_size:
            raise ValueError(f"Data size mismatch! Expected {expected_size} floats, but got {len(data)}.")

        # Reshape the data into vectors
        vectors = data.reshape(num_vectors, vector_size)
        print(f"Vectors reshaped to: {vectors.shape}")

        # Print a few vectors for inspection
        print("First 3 vectors:")
        print(vectors[:3])

    return vectors

def load_label_pair_frequency_pickle(input_path):
    """
    Load the label pair frequency dictionary from a Pickle file.
    """
    with open(input_path, 'rb') as f:
        label_pair_freq = pickle.load(f)
    print(f"Label pair frequency loaded from {input_path} (Pickle format)")
    return label_pair_freq

def load_query_file(file_path, data_type=np.float32):
    """
    Load query vectors from a binary file.
    """
    with open(file_path, 'rb') as f:
        header = f.read(8)
        num_vectors, vector_size = struct.unpack('ii', header)
        data = np.fromfile(f, dtype=data_type, count=num_vectors * vector_size)
        vectors = data.reshape(num_vectors, vector_size)
    print(f"Query vectors loaded from {file_path} (Binary format)")
    return vectors

def load_query_labels(label_path):
    """
    Load query labels from a file.
    """
    with open(label_path, 'r') as f:
        labels = [line.strip().split('&') for line in f.readlines()]
    print(f"Query labels loaded from {label_path}")
    # return np.array(labels)
    return labels

def save_selected_vectors_to_bin(output_bin_file, selected_vectors, data_type=np.float32):
    """
    Save the selected vectors to a binary file.
    The first 8 bytes store the number of vectors and their dimension.
    """
    num_vectors = len(selected_vectors)
    vector_size = len(selected_vectors[0]) if num_vectors > 0 else 0

    with open(output_bin_file, 'wb') as f:
        # Write the header (number of vectors and dimension)
        f.write(struct.pack('ii', num_vectors, vector_size))
        # Write the vector data
        for vector in selected_vectors:
            if (data_type == np.float32):
                f.write(struct.pack(f'{vector_size}f', *vector))

            if (data_type == np.uint8):
                f.write(struct.pack(f'{vector_size}B', *vector))
    print(f"Selected vectors saved to {output_bin_file} (Binary format)")


def save_selected_labels_to_file(output_label_file, selected_labels):
    """
    Save the selected labels to a text file.
    Each line contains a label pair joined by '&'.
    """
    with open(output_label_file, 'w') as f:
        for label_pair in selected_labels:
            f.write('&'.join(label_pair) + '\n')
    print(f"Selected labels saved to {output_label_file} (Text format)")


def select_query_vectors_and_labels(query_vectors_file, query_labels_file, label_pair_freq_file, output_bin_file, output_label_file, data_type = np.float32):
    """
    Select query vectors and labels based on the label pair frequency and save them to files.
    """
    label_pair_freq = load_label_pair_frequency_pickle(label_pair_freq_file)
    print(f"Label pair frequency list size: {len(label_pair_freq)}")
    print(f"Label pair frequency list (first 5): {label_pair_freq[:5]}")  # Print first 5 entries for inspection

    # Convert the list of tuples to a dictionary for easier lookup
    label_pair_freq_dict = dict(label_pair_freq)

    query_vectors = load_query_file(query_vectors_file, data_type)
    query_labels = load_query_labels(query_labels_file)
    print(f"Query vectors shape: {query_vectors.shape}")
    print(f"Query labels shape: {len(query_labels)}")
    
    selected_vectors = []
    selected_labels = []

    for i in range(len(query_labels)):
        
        if len(query_labels[i]) < 2 :
            selected_vectors.append(query_vectors[i])
            selected_labels.append(query_labels[i])
            continue
        
        if query_labels[i][0] == query_labels[i][1]:
            print(f"Skipping index {i} due to same labels: {query_labels[i]}")
            continue
        label_pair = (query_labels[i][0], query_labels[i][1])
        if label_pair in label_pair_freq_dict:
            freq = label_pair_freq_dict[label_pair]
            print(f"Label pair {label_pair} has frequency {freq}")
            # Add the vector and label to the selected lists
            selected_vectors.append(query_vectors[i])
            selected_labels.append(query_labels[i])
    
    print(f"Selected {len(selected_vectors)} vectors and labels based on frequency.")
    print(f"Selected vectors shape: {np.array(selected_vectors).shape}")

    # Save the selected vectors to a binary file
    save_selected_vectors_to_bin(output_bin_file, selected_vectors, data_type)

    # Save the selected labels to a text file
    save_selected_labels_to_file(output_label_file, selected_labels)



# def select_query_vectors_and_labels(query_vectors_file, query_labels_file, label_pair_freq_file, output_query_bin_file, output_query_label_file):
#     """
#     Select query vectors and labels based on the label pair frequency.
#     """
#     label_pair_freq = load_label_pair_frequency_pickle(label_pair_freq_file)
#     print(f"Label pair frequency list size: {len(label_pair_freq)}")
#     print(f"Label pair frequency list (first 5): {label_pair_freq[:5]}")  # Print first 5 entries for inspection

#     # Convert the list of tuples to a dictionary for easier lookup
#     label_pair_freq_dict = dict(label_pair_freq)

#     query_vectors = load_query_file(query_vectors_file)
#     query_labels = load_query_labels(query_labels_file)
#     print(f"Query vectors shape: {query_vectors.shape}")
#     print(f"Query labels shape: {query_labels.shape}")
    
#     with open(query_labels_file, 'r') as f:
#     for i in range(len(query_labels)):
#         if query_labels[i][0] == query_labels[i][1]:
#             print(f"Skipping index {i} due to same labels: {query_labels[i]}")
#             continue
#         label_pair = (query_labels[i][0], query_labels[i][1])
#         if label_pair in label_pair_freq_dict:
#             freq = label_pair_freq_dict[label_pair]
#             print(f"Label pair {label_pair} has frequency {freq}")
#             # Select the vector and label
#             selected_vector = query_vectors[i]
#             selected_label = query_labels[i]
#             print(f"Selected vector: {i}, Selected label: {selected_label}")


def main():
    # # Example usage
    # query_labels_file = '/data/wikipedia/query/query_label_reduced_5k_double.txt'
    # query_vectors_file = '/data/wikipedia/query/wiki_query_vector_5k.bin'
    # label_pair_freq_text_file = '/data/wikipedia/analysis/label_pair_frequency.txt'
    # label_pair_freq_file = '/data/wikipedia/analysis/label_pair_frequency.pkl'
    # output_query_bin_file = '/data/wikipedia/query/query_vectors_threshold_100000.bin'
    # output_query_label_file = '/data/wikipedia/query/query_labels_threshold_100000.txt'
    
    query_labels_file = '/home/t-asutradhar/yfcc/filtered_query_labels.txt'
    query_vectors_file = '/home/t-asutradhar/yfcc/filtered_query.bin'
    label_pair_freq_file = '/home/t-asutradhar/label_pair_frequency_filtered_queries.pkl'
    output_query_bin_file = '/home/t-asutradhar/yfcc/filtered_query_threshold_5k_double_single.bin'
    output_query_label_file = '/home/t-asutradhar/yfcc/filtered_query_labels_threshold_5k_double_single.txt'
    data_type = np.uint8
    

    select_query_vectors_and_labels(query_vectors_file, query_labels_file, label_pair_freq_file, output_query_bin_file, output_query_label_file, data_type)
    vectors = read_bin_file(output_query_bin_file, data_type)
    
if __name__ == "__main__":
    main()