// authors: Siddharth Gollapudi, Varun Sivashankar
// emails: t-gollapudis@microsoft.com, t-varunsi@microsoft.com

#include <boost/program_options.hpp>
#include <bits/types/struct_iovec.h>
#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <tuple>

#include "index.h"


namespace po = boost::program_options;

// oh god
#define   LIKELY(condition) __builtin_expect(static_cast<bool>(condition), 1)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

// custom types (for readability)
typedef std::string label;
typedef tsl::robin_set<label> label_set;

// structs for returning multiple items from a function
typedef std::tuple<std::vector<label_set>, tsl::robin_map<label, _u32>, label_set> parse_label_file_return_values;


/*
 * Inline function to handle command line parsing.
 *
 * Arguments are merely the inputs from the command line.
 */
inline size_t handle_args(int argc, char **argv, std::string &data_type, std::string &input_data_path, 
													std::string &final_index_path_prefix, std::string &label_data_path, std::string &universal_label, 
													unsigned &num_threads, unsigned &R, unsigned &L, unsigned &stitched_R, float &alpha) {
	po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("data_path",
                       po::value<std::string>(&input_data_path)->required(),
                       "Input data file in bin format");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&final_index_path_prefix)->required(),
                       "Path prefix for saving index file components");
    desc.add_options()("max_degree,R",
                       po::value<uint32_t>(&R)->default_value(64),
                       "Maximum graph degree");
    desc.add_options()(
        "Lbuild,L", po::value<uint32_t>(&L)->default_value(100),
        "Build complexity, higher value results in better graphs");
		desc.add_options()(
        "stitched_R", po::value<uint32_t>(&L)->default_value(100),
        "Degree to prune final graph down to");
    desc.add_options()(
        "alpha", po::value<float>(&alpha)->default_value(1.2f),
        "alpha controls density and diameter of graph, set 1 for sparse graph, "
        "1.2 or 1.4 for denser graphs with lower diameter");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()(
        "label_file",
        po::value<std::string>(&label_data_path)->default_value(""),
        "Input label file in txt format if present");
    desc.add_options()(
        "universal_label",
        po::value<std::string>(&universal_label)->default_value(""),
        "Universal label, if using it, only in conjunction with labels_file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }
	
	return 1;
}

/*
 * Parses the label datafile, which has the following line format:
 * 						point_id -> \t -> comma-separated labels
 *
 * Returns three objects in via std::tuple:
 * 1. the label universe as a set
 */
parse_label_file_return_values parse_label_file (std::string label_data_path, std::string universal_label) {
	std::ifstream label_data_stream(label_data_path);
  std::string   line, token;
  unsigned      line_cnt = 0;

	// allows us to reserve space for the points_to_labels vector
	while (std::getline(label_data_stream, line))
    line_cnt++;
	label_data_stream.clear();
  label_data_stream.seekg(0, std::ios::beg);

	// values to return
	std::vector<label_set> point_ids_to_labels(line_cnt);
	tsl::robin_map<label, _u32> labels_to_number_of_points;
	label_set all_labels;
 	
	std::vector<_u32> points_with_universal_label;
	while (std::getline(label_data_stream, line)) {
		std::istringstream       current_point_and_labels(line);
    label_set current_labels;

		// get point id
    getline(current_point_and_labels, token, '\t');
		_u32 point_id = (_u32) std::stoul(token);

		// parse comma separated labels
		getline(current_point_and_labels, token, '\t');
		std::istringstream current_labels_comma_separated(token);
		bool current_universal_label_check = false;
		while (getline(current_labels_comma_separated, token, ',')) {
			token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
			token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());

			// if token is empty, there's no labels for the point
			if (token == universal_label) {
				points_with_universal_label.push_back(point_id);
				current_universal_label_check = true;
			} else {
				all_labels.insert(token);
				current_labels.insert(token);
				labels_to_number_of_points[token]++;
			}
		}

		if (current_labels.size() <= 0 && !current_universal_label_check) {
			std::cerr << "Error: " << point_id << " has no labels." << std::endl;
			exit(-1);
		} 
		point_ids_to_labels[point_id] = current_labels;
	}

	// for every point with universal label, set its label set to all labels
	// also, increment the count for number of points a label has
	for (const auto &point_id : points_with_universal_label) {
		point_ids_to_labels[point_id] = all_labels;
		for (const auto &label : all_labels)
			labels_to_number_of_points[label]++;
	}

	std::cout << "Identified " << all_labels.size() << " distinct label(s) \n"
						<< std::endl;

	return std::make_tuple(point_ids_to_labels, labels_to_number_of_points, all_labels);
}


/*
 * For each label, generates a file containing all vectors that have said label.
 * Utilizes platform-specific functions mmap and writev.
 *
 * Each data file is saved under the following format:
 * 		input_data_path + "_" + label
 */
template<typename T>
void generate_label_specific_vector_files(std::string input_data_path, tsl::robin_map<label, _u32> labels_to_number_of_points,
																		 std::vector<label_set> point_ids_to_labels, label_set all_labels) {
	void *memblock;
  int input_data_fd;
	_u8 SCALING = 2 * sizeof(_u32);
  struct stat sb;

	auto file_writing_timer = std::chrono::high_resolution_clock::now();
	// UNLIKELY moves the branch instr. body elsewhere, opening up cache space
	// for code that actually executes
	input_data_fd = open(input_data_path.c_str(), O_RDONLY, 0444);
	if (UNLIKELY(input_data_fd == -1))
		throw;
	if (UNLIKELY(fstat(input_data_fd, &sb)))
		throw;

	// mmap could be faster than just doing a regular sequential read?
	memblock = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, input_data_fd, 0);
	if (UNLIKELY(memblock == MAP_FAILED))
		throw;
	
	const char* begin = static_cast<char const*>(memblock);
	
	_u32 number_of_points, dimension;
	std::memcpy(&number_of_points, begin, sizeof(_u32));
	std::memcpy(&dimension, begin + sizeof(_u32), sizeof(_u32));

	// iovec necessary for using writev
	tsl::robin_map<label, iovec> label_to_vectors_map;

	// mapping from id in label file to original file
	tsl::robin_map<label, std::vector<_u32>> label_id_to_orig_id;
	for (_u32 point_id = 0; point_id < number_of_points; point_id++) {
		for (const auto &label : point_ids_to_labels[point_id]) {
			// first add metadata before adding vectors
			if (!label_to_vectors_map.count(label)) {
				iovec curr_iovec;
				curr_iovec.iov_base = malloc(SCALING + labels_to_number_of_points[label] * dimension * sizeof(T));	
				std::memcpy((char *) curr_iovec.iov_base, &labels_to_number_of_points[label], sizeof(_u32));
				curr_iovec.iov_len = sizeof(_u32);
				std::memcpy((char *) curr_iovec.iov_base + curr_iovec.iov_len, &dimension, sizeof(_u32));
				curr_iovec.iov_len += sizeof(_u32);
				label_to_vectors_map[label] = curr_iovec;
				label_id_to_orig_id[label].reserve(labels_to_number_of_points[label]);
			}
			char *current_iovec_buffer = (char *) label_to_vectors_map[label].iov_base;
			size_t *current_iovec_buffer_size = &label_to_vectors_map[label].iov_len;
			std::memcpy(current_iovec_buffer + *current_iovec_buffer_size, begin + SCALING + dimension * point_id, sizeof(T) * dimension);
			*current_iovec_buffer_size += sizeof(T) * dimension;
			label_id_to_orig_id[label].push_back(point_id);
		}
	}

	// write each label iovec to resp. file
	for (const auto &label : all_labels) {
  	int label_input_data_fd;
		std::string curr_label_input_data_path(input_data_path + "_" + label);

		label_input_data_fd = open(curr_label_input_data_path.c_str(), O_CREAT | O_WRONLY, 0644);
		if (UNLIKELY(label_input_data_fd == -1))
			throw;

		// using writev leads to one write syscall per label file
		int return_value = writev(label_input_data_fd, &label_to_vectors_map[label], 1);
		if (UNLIKELY(return_value == -1))
			throw;

		free(label_to_vectors_map[label].iov_base);
		close(label_input_data_fd);
	}

	std::chrono::duration<double> file_writing_time = std::chrono::high_resolution_clock::now() - file_writing_timer;
	std::cout << "generated " << all_labels.size() << " label-specific vector files for index building in time "
		<< file_writing_time.count() << "\n" << std::endl;

	munmap(memblock, sb.st_size);
	close(input_data_fd);
}


/*
 * Using passed in parameters and files generated from step 3, 
 * builds a vanilla diskANN index for each label.
 *
 * Each index is saved under the following path:
 *  final_index_path_prefix + "_" + label
 */
template <typename T>
void generate_label_indices(std::string input_data_path, std::string final_index_path_prefix,
														label_set all_labels, unsigned R, unsigned L, float alpha, 
														unsigned num_threads) {
	diskann::Parameters label_index_build_parameters;
	label_index_build_parameters.Set<unsigned>("R", R);
	label_index_build_parameters.Set<unsigned>("L", L);
	label_index_build_parameters.Set<unsigned>("C", 750);  // maximum candidate set size du
	label_index_build_parameters.Set<bool>("saturate_graph", 0);
	label_index_build_parameters.Set<float>("alpha", alpha);
	label_index_build_parameters.Set<unsigned>("num_threads", num_threads);

	// for each label, build an index on resp. points
	double total_indexing_time = 0.0;
	auto diskann_cout_buffer = diskann::cout.rdbuf(nullptr);
	auto std_cout_buffer = std::cout.rdbuf(nullptr);
	for (const auto &label : all_labels) {
		std::string curr_label_input_data_path(input_data_path + "_" + label);
		std::string curr_label_index_path(final_index_path_prefix + "_" + label);

		size_t number_of_label_points, dimension;
		diskann::get_bin_metadata(curr_label_input_data_path, number_of_label_points, dimension);
		diskann::Index<T> index(diskann::Metric::L2, dimension, number_of_label_points, false, false);

		auto index_build_timer = std::chrono::high_resolution_clock::now();
		index.build(curr_label_input_data_path.c_str(), number_of_label_points, label_index_build_parameters);
		std::chrono::duration<double> current_indexing_time =
      std::chrono::high_resolution_clock::now() - index_build_timer;
	
		total_indexing_time += current_indexing_time.count();
  	index.save(curr_label_index_path.c_str());
	}
	diskann::cout.rdbuf(diskann_cout_buffer);
	std::cout.rdbuf(std_cout_buffer);

	std::cout << "generated per-label indices in " << total_indexing_time << " seconds\n" << std::endl;
}


int main(int argc, char **argv) {
	// 1. handle cmdline inputs
	std::string data_type, input_data_path, final_index_path_prefix, label_data_path, universal_label;
	unsigned num_threads, R, L, stitched_R;
	float alpha;

	handle_args(argc, argv, data_type, input_data_path, final_index_path_prefix, label_data_path, universal_label, num_threads, R, L, stitched_R, alpha);

	// 2. parse label file and create necessary data structures
	std::vector<label_set> point_ids_to_labels;
	tsl::robin_map<label, _u32> labels_to_number_of_points;
	label_set all_labels;
 	
	std::tie(point_ids_to_labels, labels_to_number_of_points, all_labels) = parse_label_file(label_data_path, universal_label);

	// 3. for each label, make a separate data file
	if (data_type == "uint8")
		generate_label_specific_vector_files<uint8_t>(input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
	else if (data_type == "int8")
		generate_label_specific_vector_files<int8_t>(input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);
	else if (data_type == "float")
		generate_label_specific_vector_files<float>(input_data_path, labels_to_number_of_points, point_ids_to_labels, all_labels);

	// 4. for each created data file, create a vanilla diskANN index
	if (data_type == "uint8")
		generate_label_indices<uint8_t>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha, num_threads);
	else if (data_type == "int8")
		generate_label_indices<int8_t>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha, num_threads);
	else if (data_type == "float")
		generate_label_indices<float>(input_data_path, final_index_path_prefix, all_labels, R, L, alpha, num_threads);

	// TODO: 5. "stitch" the indices together
	// TODO: 6. run a prune on the stitched index, and save to disk
}
