import pickle

INPUT_FILE_NAME = "dann2/dann2_rnd1m_labels.txt"

def readLabelFileAndReturnLabelSet(filename):
    st = set()
    with open(filename, 'r') as file:
        labels = file.readlines()
        print(len(labels))
        for line in labels:
            for int_label in line.strip().split(','):
                if int_label == '': continue
                st.add(int_label)
            
    return st

def dumpToFile(var, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    labels = readLabelFileAndReturnLabelSet(INPUT_FILE_NAME)
    print(labels)
    print(len(labels))
    dumpToFile(labels, "label_set_dann2.pickle")
    