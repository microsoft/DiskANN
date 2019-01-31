#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

typedef std::vector<std::vector<unsigned>>    VecVec;
typedef std::vector<tsl::robin_set<unsigned>> VecSet;

void load_nsg(const char *filename, VecVec &graph, unsigned &width,
              unsigned &ep_) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *) &width, sizeof(unsigned));
  in.read((char *) &ep_, sizeof(unsigned));
  // width=100;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    graph.push_back(tmp);
  }
}

void nsg_bfs(const VecVec &nsg, const unsigned ep_, VecSet &bfs_order) {
  tsl::robin_set<unsigned> *cur_level;
  tsl::robin_set<unsigned> *prev_level;
  prev_level->insert(ep_);
  unsigned level = 0;

  // set visited array to 0
  std::vector<bool> visited(nsg.size(), false);
  while(true){
    // clear state
    cur_level->clear();

    // select candidates
    for(auto id : *prev_level){
      for(const auto &nbr : nsg[id]){
        if(!visited[nbr]){
          cur_level->insert(nbr);
          visited[nbr] = true;
        }
      }
    }

    if(cur_level->empty()){
      break;
    }

    std::cerr << "Level #" << level << " : " << cur_level->size() << " nodes" << std::endl;
    
    // create a new set
    tsl::robin_set<unsigned> add(cur_level->size());
    add.insert(cur_level->begin(), cur_level->end());
    bfs_order.push_back(add);

    // swap cur_level and prev_level, increment level
    prev_level->clear();
    std::swap(prev_level, cur_level);
    level++;
  }

  // assert(visited[i] == true) for all nodes
  for(const auto val : visited){
    assert(val);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << argv[0] << " nsg_path" << std::endl;
    exit(-1);
  }

  VecVec nsg;
  VecSet bfs_order;
  unsigned ep_, width;
  load_nsg(argv[1], nsg, width, ep_);
  
  nsg_bfs(nsg, ep_, bfs_order);

  return 0;
}
