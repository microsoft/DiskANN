#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
struct Neighbor{
  unsigned id;
  float dist;
  bool flag;
};
int main(int argc, char** argv){
  if(argc != 3){std::cout << "./exec kgraph ivec_file" << '\n';return 0;}
  ifstream in(argv[1], ios::binary);
  ofstream out(argv[2], ios::binary);

  char magic[0];
  in.read(magic, 8);
  magic[8]=0;
  std::cout << magic << '\n';
  unsigned version;
  in.read((char*)&version, sizeof(unsigned));
  std::cout << version << '\n';
  unsigned format;
  in.read((char*)&format, sizeof(unsigned));
  std::cout << format << '\n';
  unsigned N;
  in.read((char*)&N, sizeof(unsigned));
  std::cout << N << '\n';
  for(unsigned i=0; i<N; i++){
    unsigned M;
    in.read((char*)&M, sizeof(unsigned));
    //std::cout << M << '\n';
    unsigned K;
    in.read((char*)&K, sizeof(unsigned));
    //std::cout << K << '\n';
    std::vector<Neighbor> vn(K);
    in.read((char*)(vn.data()), sizeof(Neighbor) * K);
    out.write((char*)&K, sizeof(unsigned));
    for(unsigned j=0; j<K; j++){
      out.write((char*)&(vn[j].id), sizeof(unsigned));
    }
  }
  return 0;
}
