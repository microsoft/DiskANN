#include <efanna2e/index_nsg.h>
#include <efanna2e/util.h>
#include <omp.h>

void load_data(char* filename, float*& data, unsigned& num,
	unsigned& dim) {  // load data with sift10K pattern
	std::ifstream in(filename, std::ios::binary);
	if (!in.is_open()) {
		std::cout << "open file error" << std::endl;
		exit(-1);
	}
	in.read((char*)&dim, 4);
	std::cout << "data dimension: " << dim << std::endl;
	in.seekg(0, std::ios::end);
	std::ios::pos_type ss = in.tellg();

	size_t fsize = (size_t)ss;
	num = (unsigned)(fsize / (dim + 1) / 4);
	std::cout << "Reading " << num << " points" << std::endl;
	data = new float[(size_t)num * (size_t)dim];

	in.seekg(0, std::ios::beg);
	for (size_t i = 0; i < num; i++) {
		in.seekg(4, std::ios::cur);
		in.read((char*)(data + i * dim), dim * 4);
	}
	in.close();
}

void save_result(char* filename, unsigned* results, unsigned nd, unsigned nr) {
	std::ofstream out(filename, std::ios::binary | std::ios::out);

	for (unsigned i = 0; i < nd; i++) {
		out.write((char*)&nr, sizeof(unsigned));
		out.write((char*)(results + i * nr), nr * sizeof(unsigned));
	}
	out.close();
}

int main(int argc, char** argv) {
	if (argc != 8) {
		std::cout << argv[0] << " data_file query_file nsg_path search_L search_K "
			"result_path BeamWidth"
			<< std::endl;
		exit(-1);
	}
	float*   data_load = NULL;
	unsigned points_num, dim;
	// load_data(argv[1], data_load, points_num, dim);
	efanna2e::load_Tvecs<float>(argv[1], data_load, points_num, dim);
	float*   query_load = NULL;
	unsigned query_num, query_dim;
	// load_data(argv[2], query_load, query_num, query_dim);
	efanna2e::load_Tvecs<float>(argv[2], query_load, query_num, query_dim);
	assert(dim == query_dim);
	std::cout << "Base and query data loaded" << std::endl;

	unsigned L = (unsigned)atoi(argv[4]);
	unsigned K = (unsigned)atoi(argv[5]);
	int      beam_width = atoi(argv[7]);

	if (L < K) {
		std::cout << "search_L cannot be smaller than search_K!" << std::endl;
		exit(-1);
	}

	data_load = efanna2e::data_align(data_load, points_num, dim);  
	query_load = efanna2e::data_align(query_load, query_num, query_dim);
	std::cout << "Data Aligned" << std::endl;

	std::vector<unsigned> picked;
	efanna2e::IndexNSG small_index(dim, 0, efanna2e::L2, nullptr);
	small_index.LoadSmallIndex(argv[3], picked);  // to load higher level NSG
	std::cout << "Small idex loaded" << std::endl;
	if (picked.size() != small_index.GetSizeOfDataset()) {
		std::cerr << "Small Index size mismatch" << std::endl;
		exit(-1);
	}
	float *small_data = new float[picked.size() * (size_t)dim];
	int i = 0;
	for (auto iter : picked)
		memcpy((char*)(small_data + (i++)*dim),
		(char*)(data_load + iter * dim), sizeof(float)*dim);

	efanna2e::IndexNSG big_index(dim, points_num, efanna2e::L2, nullptr);
	big_index.Load(argv[3]);  // to load NSG
	std::cout << "Big idex loaded" << std::endl;
	big_index.set_start_node(picked[small_index.get_start_node()]);

	efanna2e::Parameters paras;
	paras.Set<unsigned>("L_search", L);
	paras.Set<unsigned>("P_search", L);

	auto s = std::chrono::high_resolution_clock::now();

	long long total_hops = 0;
	long long total_cmps = 0;
	unsigned* res = new unsigned[(size_t)query_num * K];

#pragma omp parallel for schedule(static, 1000)
	for (unsigned i = 0; i < query_num; i++) {
		small_index.Search(query_load + i * dim, small_data,
			K, paras, res + ((size_t)i) * K);

	/*	auto ret = big_index.BeamSearch(query_load + i * dim, data_load, 
			K, paras, res + ((size_t)i) * K, beam_width);

#pragma omp atomic
		total_hops += ret.first;
#pragma omp atomic
		total_cmps += ret.second;*/
	}

	auto e = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = e - s;
	std::cout << "search time: " << diff.count() << "\n";

	std::cout << "Average hops: " << (float)total_hops / (float)query_num << std::endl
		<< "Average cmps: " << (float)total_cmps / (float)query_num << std::endl;

	save_result(argv[6], res, query_num, K);

	delete[] small_data;
	delete[] data_load;
	delete[] res;

	return 0;
}
