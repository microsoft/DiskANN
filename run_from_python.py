import subprocess
import os

# Path to the executable and its arguments
exe_path = r"C:\Users\ameyv\MSR\DiskANN\x64\Debug\build_memory_index.exe"
#exe_path = r"C:\Users\ameyv\MSR\DiskANN\x64\Release\build_disk_index.exe"
#exe_path = r"C:\Users\ameyv\MSR\DiskANN\x64\Release\gen_random_slice.exe"

coomand_type = 1

if coomand_type == 1:
    data_path = r"C:\Users\ameyv\MSR\sift_base.bin"
    index_path_prefix = r"C:\Users\ameyv\MSR\sift_memory_index_R64_L100"
    arguments = [
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", data_path,
        "--index_path_prefix", index_path_prefix,
        "-R", "64",
        "-L", "100"
    ]
elif coomand_type == 2:
    memory = 1
    data_path = r"C:\Users\ameyv\MSR\sift_base.bin"
    index_path_prefix = r"C:\Users\ameyv\MSR\sift_disk_index"
    arguments = [
        "--data_type", "float",
        "--dist_fn", "l2",
        "--data_path", data_path,
        "--index_path_prefix", index_path_prefix,
        "-R", "16",
        "-L", "25",
        "-B", "0.003",
        "-T","1",
        "-M", str(memory)
    ]
elif coomand_type == 3:
    base_file_path = r"C:\Users\ameyv\MSR\sift_base.bin"
    arguments = ["float", 
                 base_file_path, 
                 "sift_50k", 
                 "0.05"
    ]

# Constructing the complete command
command = [exe_path] + arguments

# Print the command to debug
print("Executing command:", command)

# Check if the executable exists and is a file
if not os.path.isfile(exe_path):
    print(f"Error: The executable {exe_path} does not exist or is not a file.")
else:
    # Execute the executable with arguments and capture the output
    try:
        # Run the executable with arguments and capture the output and errors
        directory = r"C:\Users\ameyv\MSR\DiskANN\results" + "\\"
        file_base = r"\log.txt"
        file_path = directory + file_base
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
        with open(directory + file_base, 'a') as log_file:  # Open the file in append mode
            result = subprocess.run(command, stdout=log_file)

    except subprocess.CalledProcessError as e:
        print(f"Program failed with return code {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Errors: {e.stderr}")

    except FileNotFoundError:
        print(f"Executable not found: {exe_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

