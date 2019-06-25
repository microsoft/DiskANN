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

// Only 4 byte data can be saved
// given data of dimension num * dim in row major, save it as fvecs with new_dim
// dimension by adding 0s or truncating it to the first new_dim dimensions
template <class saveType>
void save_Tvecs(const char* filename, saveType*& data, size_t num, size_t dim,
		size_t new_dim = 0) {
	char* buf;
	int fd;
	fd = open(filename, O_RDWR | O_CREAT | O_TRUNC, 0666);
	if (!(fd > 0)) {
		std::cerr << "ERROR: " << filename
			  << " not found. Program will stop now." << std::endl;
		assert(false);
	}

	if (new_dim == 0) new_dim = dim;
	if (new_dim < dim)
		std::cout << "save_fvecs/ivecs "<<filename<<". Current Dimension: "<<dim<<". New Dimension: First "<<new_dim<<" columns. "<<std::flush;
	else if (new_dim > dim)
		std::cout << "save_fvecs/ivecs "<<filename<<". Current Dimension: "<<dim<<". New Dimension: "<<new_dim<<" (added columns with 0 entries). "<<std::flush;
	else std::cout << "save_fvecs/ivecs "<<filename<<". Dimension: "<<dim<<". "<<std::flush;

	std::cout<<"# Points: "<<num<<".."<<std::flush;

	size_t fileSize = num * (4 * (new_dim + 1));
	ftruncate(fd, fileSize);
	buf = (char*)mmap(NULL, fileSize, PROT_WRITE, MAP_SHARED, fd, 0);
	assert(buf);
	uint32_t x = new_dim;

	float* zeros = new float[new_dim];
	for (size_t i = 0; i < new_dim; i++) zeros[i] = 0;

#pragma omp parallel for schedule(static, 65536)
	for (size_t i = 0; i < num; i++) {
		char* reader = buf + (i * (4 * new_dim + 4));
		std::memcpy(reader, &x, sizeof(uint32_t));
		reader = reader + 4;
		if (new_dim <= dim)
			std::memcpy(reader, data + i * dim,
				    new_dim * sizeof(float));
		else {
			std::memcpy(reader, zeros, new_dim * sizeof(float));
			std::memcpy(reader, data + i * dim,
				    dim * sizeof(float));
		}
	}
	int val = munmap(buf, fileSize);
	close(fd);
	std::cout<<"done."<<std::endl;
}

// plain saves data as npts X ndims array into filename
template<typename T>
void save_Tvecs_plain(const char* filename, T* data, size_t npts, size_t ndims){
	typedef uint64_t _u64;
	typedef uint32_t _u32;
	std::string fname(filename);

	// create cached ofstream with 64MB cache
	cached_ofstream writer(fname, 64 * 1048576);

	unsigned dims_u32 = (unsigned) ndims;

	// start writing
	for(_u64 i=0;i<npts;i++){
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

// Only fvecs (4 byte data) can be loaded
// given fvecs file, allocate memory for data and store fvecs fils as num *
// new_dim in row major. Appropriately truncate or expand 0s if dimension !=
// new_dim.

template <class loadType>
void load_Tvecs(const char* filename, loadType*& data, size_t& num,
		size_t& dim, size_t new_dim = 0) {
	char* buf;
	int fd;
	fd = open(filename, O_RDONLY);
	if (!(fd > 0)) {
		std::cerr << "ERROR: " << filename
			  << " not found. Program will stop now." << std::endl;
		assert(false);
	}
	struct stat sb;
	assert(fstat(fd, &sb) == 0);
	off_t fileSize = sb.st_size;
	assert(sizeof(off_t) == 8);

	buf = (char*)mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
	assert(buf);
	// size_t x=4;
	uint32_t file_dim;
	std::memcpy(&file_dim, buf, 4);
	dim = file_dim;
	if (new_dim == 0) new_dim = dim;

	if (new_dim < dim)
		std::cout << "load_fvecs/ivecs "<<filename<<". Current Dimension: "<<dim<<". New Dimension: First "<<new_dim<<" columns. "<<std::flush;
	else if (new_dim > dim)
		std::cout << "load_fvecs/ivecs "<<filename<<". Current Dimension: "<<dim<<". New Dimension: "<<new_dim<<" (added columns with 0 entries). "<<std::flush;
	else std::cout << "load_fvecs/ivecs "<<filename<<". Dimension: "<<dim<<". "<<std::flush;


	float* zeros = new float[new_dim];
	for (size_t i = 0; i < new_dim; i++) zeros[i] = 0;

	num = (size_t)(fileSize / (dim + 1) / 4);
	data = new loadType[(size_t)num * (size_t)new_dim];

	std::cout<<"# Points: "<<num<<".."<<std::flush;

#pragma omp parallel for schedule(static, 65536)
	for (size_t i = 0; i < num; i++) {
		char* reader = buf + (i * (dim + 1) * 4);
		std::memcpy(data + (i * new_dim), zeros,
			    new_dim * sizeof(float));
		if (new_dim > dim)
			std::memcpy(data + (i * new_dim), (reader + 4),
				    dim * sizeof(float));
		else
			std::memcpy(data + (i * new_dim), (reader + 4),
				    new_dim * sizeof(float));
	}

	int val = munmap(buf, fileSize);
	close(fd);
	std::cout<<"done."<<std::endl;
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
