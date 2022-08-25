// authors: Siddharth Gollapudi, Varun Sivashankar
// emails: t-gollapudis@microsoft.com, t-varunsi@microsoft.com

#include <boost/program_options.hpp>
#include <omp.h>

#include "index.h"


namespace po = boost::program_options;

// custom types (for readability)
typedef std::string label;


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

int main(int argc, char **argv) {
	// 1. handle cmdline inputs
	std::string data_type, input_data_path, final_index_path_prefix, label_data_path, universal_label;
	unsigned num_threads, R, L, stitched_R;
	float alpha;

	handle_args(argc, argv, data_type, input_data_path, final_index_path_prefix, label_data_path, universal_label, num_threads, R, L, stitched_R, alpha);

	// TODO: 2. parse label file and create necessary data structures
	// TODO: 3. for each label, make a separate data file
	// TODO: 4. for each created data file, create a vanilla diskANN index
	// TODO: 5. "stitch" the indices together
	// TODO: 6. run a prune on the stitched index, and save to disk
}
