
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>


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

typedef unsigned long long ull;

int main(int argc, char** argv)
{
  if(argc!=3){std::cout<< argv[0] <<" data_file1 data_file2"<<std::endl; exit(-1);}
  int* gold_std = NULL;
  int* our_results = NULL;
  ull points_num;
  unsigned dim_gs;
  unsigned dim_or;
  load_data(argv[1], gold_std, points_num, dim_gs);
  load_data(argv[2], our_results, points_num, dim_or);
  ull recall  =0;
  bool* all_recall = new bool[points_num];
  for (unsigned i = 0;i < points_num; i++)
	  all_recall[i] = false;

  std::cout<<"calculating recall "<<dim_gs<<" at "<<dim_or<<std::endl;
  unsigned mind = dim_gs;
  if(dim_or < mind)
  {
	  std::cout<<"ground truth has size "<< dim_gs<<" and our set has only "<< dim_or<<" points. exiting \n";	
	  return(1);
  }
		

    
	bool* this_point = new bool[dim_gs];
	for (ull i = 0; i < points_num; i++)
	{
		for(unsigned j=0;j<dim_gs;j++)
			this_point[j] = false;

		bool this_correct = true;
		for (ull j1 = 0; j1 < dim_gs; j1++)
			for (ull j2 = 0; j2 < dim_or; j2++)
				if (gold_std[i*(ull)dim_gs + j1] == our_results[i*(ull)dim_or + j2]) 
					this_point[j1] = true;
		for(unsigned j1 = 0; j1 < dim_gs; j1++)
			if(this_point[j1] == false)
			{
				this_correct = false;
				break;
			}
		if(this_correct == true)
			recall++;

	}
	

double avg_recall = (recall*1.0)/(points_num*1.0);
std::cout <<"avg. recall "<< dim_gs <<" at "<< dim_or<<" is "<< avg_recall<<" \n";
} 
