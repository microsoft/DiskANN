import pickle

INPUT_FILE_NAME = "dann2/dann2_rnd1m_labels.txt" # n = 980312 for wiki, 1109275 for dann2
OUTPUT_FILE_NAME = "wiki_rnd1m/dann2_rnd1m_caps_labels.txt"
LABEL_SET_FILE = "label_set_dann2.pickle"

def readWikiLabelFile(filename, n=1109275):
    all_labels = []
    with open(filename, 'r') as file:
        for _ in range(n):
            labels = file.readline()
            labels = filter(lambda x: x != '', labels.strip().split(','))
            all_labels.append(set(labels))
            if(_%100000==0):
                print(str(_) + " done")
    return all_labels

def writeLabelsInCapsFormat(filename, labels, label_map, n=1109275):
    labels_count = len(label_map)
    with open(filename, 'w') as file:
        file.write(str(n) + ' ' + str(labels_count) + '\n')
        for i in range(n):
            # print_labels = [getCapsFormat(1 if j in labels[i] else 0, j) for j in range(labels_count)]
            print_labels = [0 for j in range(labels_count)]
            for label in labels[i]:
                print_labels[label_map[label]] = 1
            print_labels = [getCapsFormat(label, j) for j, label in enumerate(print_labels)]
            file.write(' '.join(print_labels) + '\n')
            if(i%100000==0):
                print(str(i) + " done")
        
def getCapsFormat(label, position):
    return str(position) + '_' + str(label)

def loadFromFile(filename):
    result = None
    with open(filename, 'rb') as handle:
        result = pickle.load(handle)
    return result

def convertToLabelMap(label_set):
    label_map = {}
    for i, label in enumerate(label_set):
        label_map[label] = i
    return label_map


if __name__ == "__main__":
    label_set = loadFromFile(LABEL_SET_FILE)
    label_map = convertToLabelMap(label_set)
    labels = readWikiLabelFile(INPUT_FILE_NAME)
    writeLabelsInCapsFormat(OUTPUT_FILE_NAME, labels, label_map)