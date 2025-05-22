import struct
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle


def load_query_labels(label_path):
    """
    Load query labels from a file.
    """
    with open(label_path, 'r') as f:
        labels = [line.strip().split('&') for line in f.readlines()]
    # return np.array(labels)
    return labels


def load_query_file(file_path):
    """
    Load query vectors from a binary file.
    """
    with open(file_path, 'rb') as f:
        header = f.read(8)
        num_vectors, vector_size = struct.unpack('ii', header)
        data = np.fromfile(f, dtype=np.float32, count=num_vectors * vector_size)
        vectors = data.reshape(num_vectors, vector_size)
    return vectors


def load_labels(label_path):
    """
    Load vector labels from a file.
    """
    with open(label_path, 'r') as f:
        labels = [line.strip().split(',') for line in tqdm(f, desc="Loading labels", dynamic_ncols=True)]
    return labels


def preprocess_vector_labels(vector_labels):
    """
    Preprocess vector labels into a mapping of label -> set of vector indices.
    """
    label_to_vectors = defaultdict(set)
    for i, labels in enumerate(vector_labels):
        for label in labels:
            label_to_vectors[label].add(i)
    return label_to_vectors


def get_label_pair_frequency(query_labels, vector_labels):
    """
    Get the frequency of query label pairs in the vector labels.
    """
    # Preprocess vector labels for efficient lookup
    label_to_vectors = preprocess_vector_labels(vector_labels)

    # Calculate label pair frequencies
    label_pair_freq = {}
    for labels in tqdm(query_labels, desc="Processing query labels", dynamic_ncols=True):
        if len(labels) < 2:
            # Skip queries with fewer than 2 labels
            continue

        l1, l2 = labels[:2]  # Unpack the first two labels
        if l1 == l2:
            continue
        if (l1, l2) not in label_pair_freq:
            label_pair_freq[(l1, l2)] = 0

        # Get the intersection of vectors containing l1 and l2
        if l1 in label_to_vectors and l2 in label_to_vectors:
            common_vectors = label_to_vectors[l1] & label_to_vectors[l2]
            label_pair_freq[(l1, l2)] = len(common_vectors)

    return label_pair_freq


def save_label_pair_frequency(sorted_label_pair_freq, output_path, threshold=None):
    """
    Save the label pair frequency to a file.
    """
    # If a threshold is provided, limit the number of pairs saved
    if threshold:
        sorted_label_pair_freq = sorted_label_pair_freq[:threshold]

    with open(output_path, 'w') as f:
        for pair, freq in sorted_label_pair_freq:
            f.write(f"{pair}: {freq}\n")
            
def save_label_pair_frequency_pickle(label_pair_freq, output_path):
    """
    Save the label pair frequency dictionary to a file using Pickle.
    """
    with open(output_path, 'wb') as f:
        pickle.dump(label_pair_freq, f)
    print(f"Label pair frequency saved to {output_path} (Pickle format)")


def load_label_pair_frequency_pickle(input_path):
    """
    Load the label pair frequency dictionary from a Pickle file.
    """
    with open(input_path, 'rb') as f:
        label_pair_freq = pickle.load(f)
    print(f"Label pair frequency loaded from {input_path} (Pickle format)")
    return label_pair_freq


def main():
    # Parameters
    # query_label_path = '/data/wikipedia/query/query_label_reduced_5k_double.txt'
    # vector_label_path = '/data/wikipedia/cleaned_lemmatized_reduced_wiki_label.txt'
    # output_path_txt = '/data/wikipedia/analysis/label_pair_frequency.txt'
    # output_path_pickle = '/data/wikipedia/analysis/label_pair_frequency.pkl'
    # threshold = 100000
    
    query_label_path = '/home/t-asutradhar/yfcc/filtered_query_labels.txt'
    vector_label_path = '/mnt/YFCC/base_1000000.u8bin.label'
    output_path_txt = '/home/t-asutradhar/label_pair_frequency_filtered_queries.txt'
    output_path_pickle = '/home/t-asutradhar/label_pair_frequency_filtered_queries.pkl'
    threshold = 5000

    # Load query labels
    query_labels = load_query_labels(query_label_path)
    # Load query labels
    query_labels = load_query_labels(query_label_path)
    print("Query labels count:", len(query_labels))  # Use len() instead of .shape
    print("Query labels (first 10):", query_labels[:10])

    # Load vector labels
    vector_labels = load_labels(vector_label_path)
    print("Vector labels count:", len(vector_labels))
    print("Vector labels (first 10):", vector_labels[:10])

    # Get label pair frequency
    label_pair_freq = get_label_pair_frequency(query_labels, vector_labels)
    
    # sort the label pair frequency
    sorted_label_pair_freq = sorted(label_pair_freq.items(), key=lambda item: item[1], reverse=True)
    print("Label pair frequency (first 10):", sorted_label_pair_freq[:10])
    sorted_label_pair_freq = [sorted_label_pair_freq[i] for i in range(len(sorted_label_pair_freq)) if sorted_label_pair_freq[i][1] > threshold]

    # Save the label pair frequency to a file
    # Save the label pair frequency to a Pickle file
    save_label_pair_frequency_pickle(sorted_label_pair_freq, output_path_pickle)

    # Optionally save to a text file for human readability
    save_label_pair_frequency(sorted_label_pair_freq, output_path_txt, threshold)

    print(f"Label pair frequency saved to {output_path_txt} and {output_path_pickle}")


if __name__ == "__main__":
    main()