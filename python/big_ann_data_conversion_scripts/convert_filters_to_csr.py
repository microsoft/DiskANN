import numpy as np
from scipy.sparse import csr_matrix

INPUT_FILE_NAME = "try.txt"
OUTPUT_FILE_NAME = "try_csr.bin"

def write_sparse_matrix(mat, fname):
    """ write a CSR matrix in the spmat format """
    with open(fname, "wb") as f:
        sizes = np.array([mat.shape[0], mat.shape[1], mat.nnz], dtype='int64')
        sizes.tofile(f)
        indptr = mat.indptr.astype('int64')
        indptr.tofile(f)
        mat.indices.astype('int32').tofile(f)
        mat.data.astype('float32').tofile(f)


def readWikiQueryLabelFile(filename, n=5000):
    all_labels = []
    with open(filename, 'r') as file:
        for _ in range(n):
            labels = file.readline()
            labels = map(int, labels.strip().split('&'))
            all_labels.append(set(labels))
            if(_%100000==0):
                print(str(_) + " done")
    return all_labels

def readWikiLabelFile(filename, n=980312):
    all_labels = []
    with open(filename, 'r') as file:
        for _ in range(n):
            labels = file.readline()
            labels = map(int, filter(lambda x: x != '', labels.strip().split(',')))
            all_labels.append(set(labels))
            if _%100000==0:
                print(str(_) + " done")
    return all_labels

def convert_to_csr(labels):
    """ convert the labels to a CSR matrix """
    cnt = 0
    rows = []
    cols = []
    for i in range(len(labels)):
        for x in labels[i]:
            cnt += 1
            rows.append(i)
            cols.append(x)
    data = np.ones(cnt)

    return csr_matrix((data, (rows, cols)), shape=(len(labels), 4000))

if __name__ == "__main__":
    labels = readWikiLabelFile(INPUT_FILE_NAME, 5)
    mat = convert_to_csr(labels)
    print(mat)
    write_sparse_matrix(mat, OUTPUT_FILE_NAME)
    