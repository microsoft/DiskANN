import pickle
import random

INPUT_FILE = "dann2/all_query_labels.txt"
OUTPUT_FILE = "dann2/all_query_labels_with_AND_diskann.txt"
LABEL_SET_FILE = "label_set_dann2.pickle"

def loadFromFile(filename):
    result = None
    with open(filename, 'rb') as handle:
        result = pickle.load(handle)
    return result

def readDanndataLabelFile(filename):
    all_labels = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            all_labels.append(line.strip())
    print(len(all_labels))
    return all_labels

def makeANDLabeledData(lables, label_list):
    for i in range(len(lables)):
        other_label = random.choice(label_list)
        while other_label == lables[i]:
            other_label = random.choice(label_list)
        lables[i] = lables[i] + '&' + other_label
    print(len(lables))

def writeToFile(filename, labels):
    with open(filename, 'w') as file:
        for label in labels:
            file.write(label + '\n')

if __name__ == "__main__":
    label_set = loadFromFile(LABEL_SET_FILE)
    label_list = list(label_set)
    labels = readDanndataLabelFile(INPUT_FILE)
    makeANDLabeledData(labels, label_list)
    writeToFile(OUTPUT_FILE, labels)
