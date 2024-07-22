import parse_common
import argparse

def get_data_type_code(data_type_name):
    if data_type_name == "float":
        return ('f', 4)
    elif data_type_name == "int8":
        return ('b', 1)
    elif data_type_name == "uint8":
        return ('B', 1)
    else:
        raise Exception("Only float, int8 and uint8 are supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a file in .bin format")
    parser.add_argument("filename", help="The vector/matrix file to parse")
    parser.add_argument("data_type", help="Type of data in the vector file. Only float, int8 and uint8 are supported.")
    parser.add_argument("output_file", help="The file to write the parsed data to")
    args = parser.parse_args()

    data_type_code, data_type_size = get_data_type_code(args.data_type)
    
    datamat = parse_common.DataMat(data_type_code, data_type_size)
    datamat.load_bin(args.filename)

    with open(args.output_file, "w") as out_file:
        for i in range(len(datamat)):
            out_file.write(str(datamat[i].tolist()) + "\n")
    
    print("Parsed " + str(len(datamat)) + " vectors from " + args.filename + " and wrote output to " + args.output_file)
    