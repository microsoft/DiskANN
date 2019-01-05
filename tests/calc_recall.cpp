
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
  std::cout<<"file size " <<fsize<<" bytes \n";
  std::cout<<"possibility 1 "<< data_size*4 + 4<<" or possibility 2 " << (unsigned long long) data_size*4 + (unsigned long long) num*4 + 4<< std::endl;  
data = new int [data_size];
  
  int *tmp_dim = new int;
  in.seekg(0,std::ios::beg);
  for(size_t i = 0; i < num; i++){
//    in.seekg(4,std::ios::cur);
    in.read((char*) tmp_dim, 4);
    in.read((char*)(data+i*dim),dim*4);
/*    if (i % 10000 == 0)
    {
	    std::cout<<"finished  "<<(i*1.0/num*1.0)*100<<" percent" << std::endl;
	    std::cout<< "vector dim " << (int) *tmp_dim  << "first and last coords " << (int) (data + i*dim)[0] <<" "<< (int) (data + i*dim)[*tmp_dim - 1]<< std::endl;     
	    std::cout <<" currently read " << i*dim*4 << "bytes out of "<< (1.0*num)*(1.0*dim)*(1.0*sizeof(float)) <<" bytes and file size is "<< fsize << "bytes and float size is "<<sizeof(float) <<" bytes \n"; 
}   */ 
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
  
  for (ull i = 0; i < points_num; i++)
    for (ull j1 = 0; j1 < dim_or; j1 ++)
      for (ull j2 = 0; j2 < dim_or; j2 ++)
	if (our_results[i*(ull)dim_or + j1] == gold_std[i*(ull)dim_gs + j2]) 
	  recall ++;


/*
  for (int i = 0; i < points_num; i++)
{
std::cout << i <<" ;";
 for (int j1 = 0; j1 < dim_or; j1 ++)
  {
std::cout << our_results[i*dim_or + j1] <<" ";
}
std::cout << "; ";
for (int j2 = 0; j2 < dim_or; j2 ++)
{

std::cout << gold_std[i*dim_gs + j2] <<" ";
}
std::cout << "\n ";
}
*/
/*
  for (unsigned long long int i = 0; i < points_num; i++)
{
std::cout << i <<" ;";
 for (int j1 = 0; j1 < dim_gs; j1 ++)
  {
std::cout << our_results[i*(unsigned long long)dim_gs + (unsigned long long)j1] <<" ";
}
std::cout << "\n ";
}
 */
double avg_recall = (recall*1.0)/(points_num*1.0);
std::cout <<"avg. recall at "<< dim_or <<" is "<< avg_recall<<" \n";
} 
