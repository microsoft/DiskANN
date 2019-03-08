
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>


#include<cstring>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <cassert>


void load_data(char* filename, int*& data, unsigned long long & num,unsigned & dim){// load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  unsigned long long int fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  unsigned long long int data_size = (unsigned long long) num * (unsigned long long) dim;
  std::cout<<"data dimension: "<<dim<<std::endl;
  std::cout<<"data num points: "<<num<<std::endl;
data = new int [data_size];
  
  int *tmp_dim = new int;
  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
//    in.seekg(4,std::ios::cur);
    in.read((char*) tmp_dim, 4);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
  std::cout <<"data loaded \n";
		
}



void open_linux_mmapped_file_handle(
    char* filename,
    float*& data,
    size_t& num,
        unsigned& dim)
{
        char* buf;
        int fd;
    fd = open(filename, O_RDONLY);
    if (!(fd > 0)) {
        std::cerr << "Data file " << filename << " not found. Program will stop now." << std::endl;
        assert(false);
    }
    struct stat sb;
    int val = fstat(fd, &sb);
    off_t fileSize = sb.st_size;
    assert(sizeof(off_t) == 8);

    buf = (char*) mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(buf);
    //size_t x=4;
    std::memcpy(&dim, buf, 4);
    num=(size_t)(fileSize/(dim+1)/4);
    data = new float[(size_t)num * (size_t)dim];


#pragma omp parallel for schedule(static, 65536)
    for (size_t i=0; i<num; i++){
        char* reader=buf+(i*(dim+1)*4);
        std::memcpy(data+(i*dim), (reader+4), dim*sizeof(float) );
    }

}


void load_fdata(char* filename, float*& data, unsigned long long & num,unsigned & dim){// load data with sift10K pattern
  std::ifstream in(filename, std::ios::binary);
  if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
  in.read((char*)&dim,4);
  in.seekg(0,std::ios::end);
  std::ios::pos_type ss = in.tellg();
  unsigned long long int fsize = (size_t)ss;
  num = (unsigned)(fsize / (dim+1) / 4);
  unsigned long long int data_size = (unsigned long long) num * (unsigned long long) dim;
  std::cout<<"data dimension: "<<dim<<std::endl;
  std::cout<<"data num points: "<<num<<std::endl;
data = new float [data_size];
  
  int *tmp_dim = new int;
  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
//    in.seekg(4,std::ios::cur);
    in.read((char*) tmp_dim, 4);
    in.read((char*)(data+i*dim),dim*4);
  }
  in.close();
  std::cout <<"data loaded \n";
		
}
 
typedef unsigned long long ull;

float calc_dist(float* vec_1, float* vec_2, size_t dim){
	float dist = 0;
	for (size_t j = 0; j<dim; j++)
	{
		dist += (vec_1[j] - vec_2[j])* (vec_1[j] - vec_2[j]);
	}
	return dist;
}

int main(int argc, char** argv)
{
  if(argc!=7 && argc!=8 ){std::cout<< argv[0] <<"ground_truth_ivecs our_ivecs base_file query_file r R  (for recall r (gt) at R (or)) our_dist_file (optional)"<<std::endl; exit(-1);}
  int* gold_std = NULL;
  int* our_results = NULL;
  ull points_num;
  unsigned r = atoi(argv[5]);
  unsigned R = atoi(argv[6]);
  unsigned dim_gs;
  unsigned dim_or;
  float* base = NULL;
  unsigned dim_;
  size_t base_num;
  ull queries_num;
  float* query = NULL;
  open_linux_mmapped_file_handle(argv[3], base, base_num,  dim_);
  load_fdata(argv[4], query, queries_num, dim_);
  load_data(argv[1], gold_std, points_num, dim_gs);
  load_data(argv[2], our_results, points_num, dim_or);
  float* dist_mat = NULL;
	if(argc == 8)
		  load_fdata(argv[7], dist_mat, points_num, dim_or);
  ull recall  =0;
  ull total_recall = 0;
  bool* all_recall = new bool[points_num];
  for (unsigned i = 0;i < points_num; i++)
	  all_recall[i] = false;

  std::cout<<"calculating recall "<<dim_gs<<" at "<<dim_or<<std::endl;
  unsigned mind = dim_gs;
  if((dim_or < dim_gs) || (!(r < dim_gs)) || (!(R < dim_or)))
  {
	  std::cout<<"ground truth has size "<< dim_gs<<" and our set has only "<< dim_or<<" points, and r and R are "<<r<<" and "<<R<<" respectively. exiting \n";	
	  return(1);
  }


		
	for (unsigned i = 0; i< 2; i++)
	{
		for(unsigned j=0;j < mind; j++)
		{
			size_t gtidx = gold_std[i*dim_gs + j];
			float gt_dist = calc_dist(query + i*dim_, base + gtidx*dim_, dim_);
			
			std::cout<<gtidx<<"("<<gt_dist<<") ";
		}
		std::cout<<"\n\n";
		for(unsigned j=0;j < mind; j++)
		{
			size_t oridx = our_results[i*dim_or + j];
			float or_dist = calc_dist(query + i*dim_, base + oridx*dim_, dim_);

			std::cout<<oridx<<"("<<dist_mat[i*dim_or+j]<<","<<or_dist<<") ";
		}
		std::cout<<"\n------------------------------------\n";
	}

	for (unsigned i = 0; i< 100; i++)
	{
		
		size_t gtidx = gold_std[i*dim_gs + 0];
		float gt_dist = calc_dist(query + i*dim_, base + gtidx*dim_, dim_);

		std::cout<<"Query i"<<"Harsha says "<<gtidx<<"("<<gt_dist<<") AND they claim ";
		size_t oridx = our_results[i*dim_or + 0];
		float or_dist = calc_dist(query + i*dim_, base + oridx*dim_, dim_);

		std::cout<<oridx<<"("<<dist_mat[i*dim_or+0]<<","<<or_dist<<") ";
		std::cout<<"\n";
	}


  bool* this_point = new bool[dim_gs];
  for (ull i = 0; i < points_num; i++)
  {
	  for(unsigned j=0;j<dim_gs;j++)
		  this_point[j] = false;

	  bool this_correct = true;
	  for (ull j1 = 0; j1 < r; j1++)
		  for (ull j2 = 0; j2 < R; j2++)
			  if (gold_std[i*(ull)dim_gs + j1] == our_results[i*(ull)dim_or + j2])
			  { 
				  if(this_point[j1] == false)
					  total_recall++;
				  this_point[j1] = true;
			  }
	  for(unsigned j1 = 0; j1 < dim_gs; j1++)
		  if(this_point[j1] == false)
		  {
			  this_correct = false;
			  break;
		  }
	  if(this_correct == true)
		  recall++;

  }


  //  double avg_recall = (recall*1.0)/(points_num*1.0);
  std::cout <<"avg. recall "<< r <<" at "<< R<<" is "
	    <<(total_recall*1.0)/points_num<<std::endl;
} 
