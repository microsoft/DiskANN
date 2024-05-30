import array
import struct


INPUT_FILE_NAME = "dann2/gt100"
OUTPUT_FILE_NAME = "dann2/gt100.ivecs"


def readBinData(filename):
    # Open the binary file in read-binary mode
    with open(filename, 'rb') as file:
        # Read the first 8 bytes and unpack them into two 4-byte integers
        N, M = struct.unpack('ii', file.read(8))
        print(N, M)
        data = []
        
        # Loop through the number of rows N
        for _ in range(N):
            # Initialize an empty list to store the data
            vector = array.array('i')
            # For each row, read M*datatype_size bytes and unpack them into datatype
            vector.fromfile(file, M)
            data.append(vector)

            if(_%100000==0):
                print(str(_) + " done")
                print("Data Length = " + str(len(data)))

    print("Data Length = " + str(len(data)))
    return data



def writeToBinaryFile(filename, data, datatype='i', dims=100):
    N = len(data)
    with open(filename, 'wb') as file:
        for i in range(N):
            file.write(struct.pack('i', dims))
            bestK = data[i][:dims]
            file.write(struct.pack(datatype*dims, *bestK))
            if(i%100000==0):
                print(str(i) + " done")


if __name__ == "__main__":
    data = readBinData(INPUT_FILE_NAME)
    writeToBinaryFile(OUTPUT_FILE_NAME, data)