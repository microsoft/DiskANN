// authors: Siddharth Gollapudi, Varun Sivashankar
// emails: t-gollapudis@microsoft.com, t-varunsi@microsoft.com

#include <boost/program_options.hpp>
#include <omp.h>
#include <tuple>

#include "index.h"


namespace po = boost::program_options;

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

	std::cout << "Identified " << all_labels.size() << " distinct label(s)"
						<< std::endl;

	return std::make_tuple(point_ids_to_labels, labels_to_number_of_points, all_labels);
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

	// TODO: 3. for each label, make a separate data file
	// TODO: 4. for each created data file, create a vanilla diskANN index
	// TODO: 5. "stitch" the indices together
	// TODO: 6. run a prune on the stitched index, and save to disk
}
