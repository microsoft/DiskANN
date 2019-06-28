#pragma once

#include <common_includes.h>
#include "cached_io.h"

struct PivotContainer {
	PivotContainer() = default;

	PivotContainer(size_t pivo_id, float pivo_dist)
	    : piv_id{pivo_id}, piv_dist{pivo_dist} {}

	bool operator<(const PivotContainer& p) const {
		return p.piv_dist < piv_dist;
	}

	bool operator>(const PivotContainer& p) const {
		return p.piv_dist > piv_dist;
	}

	size_t piv_id;
	float piv_dist;
};

// given a matrix data of dimension n * dim in row major, print the first num
// rows of data
template <class testType>
void print_test_vec(testType* data, size_t dim, size_t num) {
	for (unsigned i = 0; i < num; i++) {
		for (unsigned j = 0; j < dim; j++)
			std::cout << data[i * dim + j] << " ";
		std::cout << std::endl;
	}
}


// plain saves data as npts X ndims array into filename
template<typename T>
void save_Tvecs_plain(const char* filename, T* data, size_t npts, size_t ndims){
	std::string fname(filename);

	// create cached ofstream with 64MB cache
	cached_ofstream writer(fname, 64 * 1048576);

	unsigned dims_u32 = (unsigned) ndims;

	// start writing
	for(uint64_t i=0;i<npts;i++){
		// write dims in u32
		writer.write((char*)&dims_u32, sizeof(unsigned));

		// get cur point in data
		T* cur_pt = data + i*ndims;
		writer.write((char*)cur_pt, ndims * sizeof(T));
	}
}

// plain loads data as npts X ndims array from filename -> data
template<typename T>
void load_Tvecs_plain(const char* filename, T* &data, size_t &npts, size_t &ndims){
	typedef uint64_t _u64;
	typedef uint32_t _u32;
	std::string fname(filename);

	std::ifstream plain_reader(fname, std::ios::binary | std::ios::ate);
	_u64 fsize = plain_reader.tellg();
	unsigned dims_u32;
        plain_reader.seekg(0, std::ios::beg);
	plain_reader.read((char*)&dims_u32, sizeof(unsigned));
	plain_reader.close();
	
	_u64 vec_size = sizeof(unsigned) + dims_u32 * sizeof(T);
	npts = (size_t) fsize / (size_t) vec_size;
	std::cout << "filesize: " << fsize<< ", vec_size: " << vec_size<<", npts : " << npts << ", ndims : " << dims_u32 << '\n';
	ndims = (size_t) dims_u32;

	// create cached ifstream with 64MB cache
	cached_ifstream reader(fname, 64 * 1048576);

	// allocate storage + zero
	data = new T[npts * ndims];
	memset(data, 0, npts * ndims * sizeof(T));

	// start reading
	for(_u64 i=0;i<npts;i++){
		// read dims in u32
		reader.read((char*)&dims_u32, sizeof(unsigned));

		// get cur point in data
		T* cur_pt = data + i*ndims;
		reader.read((char*)cur_pt, ndims * sizeof(T));
	}
}

// plain loads data as npts X ndims array from filename -> data (after conversion)
template<typename InType, typename OutType>
void load_Tvecs_plain(const char* filename, OutType* &data, size_t &npts, size_t &ndims){
	typedef uint64_t _u64;
	typedef uint32_t _u32;
	std::string fname(filename);

	std::ifstream plain_reader(fname, std::ios::binary | std::ios::ate);
	_u64 fsize = plain_reader.tellg();
	unsigned dims_u32;
        plain_reader.seekg(0, std::ios::beg);
	plain_reader.read((char*)&dims_u32, sizeof(unsigned));
	plain_reader.close();
	
	_u64 vec_size = sizeof(unsigned) + dims_u32 * sizeof(InType);
	npts = (size_t) fsize / (size_t) vec_size;
	std::cout << "filesize: " << fsize<< ", vec_size: " << vec_size<<", npts : " << npts << ", ndims : " << dims_u32 << '\n';
	ndims = (size_t) dims_u32;

	// create cached ifstream with 64MB cache
	cached_ifstream reader(fname, 64 * 1048576);

	// allocate storage + zero
	data = new OutType[npts * ndims];
	InType *temp_vec = new InType[ndims];
	memset(data, 0, npts * ndims * sizeof(OutType));

	// start reading
	for(_u64 i=0;i<npts;i++){
		// read dims in u32
		reader.read((char*)&dims_u32, sizeof(unsigned));

		// get cur point in data
		OutType* cur_pt = data + i*ndims;
		reader.read((char*)temp_vec, ndims * sizeof(InType));
    for(_u64 d=0;d<ndims;d++)
       cur_pt[d] = (OutType) temp_vec[d];
	}

  delete[] temp_vec;
}

// plain loads data as npts X ndims array from filename -> data (after conversion)
template<typename InType, typename OutType>
void load_bin_plain(const char* filename, OutType* &data, size_t &npts, size_t &ndims){
	typedef uint64_t _u64;
	typedef uint32_t _u32;
	std::string fname(filename);

	cached_ifstream reader(fname, 128 * 1048576l);
	int npts_i32, ndims_i32;
	reader.read((char*)&npts_i32, sizeof(int32_t));
	reader.read((char*)&ndims_i32, sizeof(int32_t));

	npts = (size_t) npts_i32;
	ndims = (size_t) ndims_i32;
	
	std::cout << "npts: " << npts << ", ndims: " << ndims << '\n';

	// allocate storage + zero
	data = new OutType[npts * ndims];
	InType *temp_vec = new InType[ndims];
	memset(data, 0, npts * ndims * sizeof(OutType));

	// start reading
	for(_u64 i=0;i<npts;i++){
		// get cur point in data
		OutType* cur_pt = data + i*ndims;
		reader.read((char*)temp_vec, ndims * sizeof(InType));
    for(_u64 d=0;d<ndims;d++)
       cur_pt[d] = (OutType) temp_vec[d];
	}

  delete[] temp_vec;
}

void load_bvecs(const char* filename, float*& data, size_t& num, size_t& dim,
		size_t new_dim = 0); 


void debug_code(size_t* closest_center, size_t* test_set, size_t num_points,
		size_t dim, float* cur_pivot_data, float* cur_data);


template <class Datatype>
unsigned load_file_into_data (const char* filepath, Datatype* &datapath, size_t& num_rows, size_t& num_dims) {
  std::string bvecs(".bvecs");
  std::string fvecs(".fvecs");
  std::string ivecs(".ivecs");
  std::string cur_file(filepath);
int32_t return_type = 0;
  if (cur_file.find(fvecs) != std::string::npos) {
    		std::cout << "Loading file as fvecs.. " << std::flush;
    load_Tvecs_plain<float, Datatype>(filepath, datapath, num_rows, num_dims);
  return_type = 1;
  } else if (cur_file.find(bvecs) != std::string::npos)
{
    		std::cout << "Loading file as bvecs.." << std::flush;
    load_Tvecs_plain<uint8_t, Datatype>(filepath, datapath, num_rows, num_dims);
return_type = 2;
  } else if (cur_file.find(ivecs) != std::string::npos) 
{
    		std::cout << "Loading file as ivecs.." << std::flush;
    load_Tvecs_plain<uint32_t, Datatype>(filepath, datapath, num_rows, num_dims);
return_type = 3;
  }
else {
return_type = -1;
}
return return_type;
}


inline bool file_exists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}
