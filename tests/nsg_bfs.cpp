#include <algorithm>
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
  std::cout << "ep_ = " << ep_ << ", width = " << width << std::endl;
	uint64_t n_dups = 0;
  // width=100;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    assert(k <= width);
    if (k == 0) {
      std::cerr << "Detached node - id= " << graph.size() << std::endl;
    }
    if (in.eof())
      break;
    std::vector<unsigned> tmp(k);
		tsl::robin_set<unsigned> tmp_set;
    in.read((char *) tmp.data(), k * sizeof(unsigned));
		for(unsigned id : tmp)
			tmp_set.insert(id);
		if ((graph.size() % 1000000) == 0)
			std::cout << graph.size() << " nodes read" << std::endl;
		/*
		if (tmp_set.size() != k)
			std::cout << "duplicate nbrs: " << tmp_set.size() << " vs " << k << std::endl;
		*/
		n_dups += (tmp.size() - tmp_set.size());
    graph.push_back(tmp);
  }
	std::cout << "Total # of dups: " << n_dups << ", avg # of dups: " << (double)n_dups / (double)graph.size() << std::endl;
  std::cout << "Total #nodes = " << graph.size() << std::endl;
}

void nsg_bfs(const VecVec &nsg, const unsigned start_node, VecSet *bfs_order,
             bool *visited) {
  tsl::robin_set<unsigned> *cur_level = new tsl::robin_set<unsigned>();
  tsl::robin_set<unsigned> *prev_level = new tsl::robin_set<unsigned>();
  prev_level->insert(start_node);
  visited[start_node] = true;
  unsigned level = 0;
  unsigned nsg_size = nsg.size();
  while (true) {
    // clear state
    cur_level->clear();

    // select candidates
    for (auto id : *prev_level) {
      for (const auto &nbr : nsg[id]) {
        if (nbr >= nsg_size) {
          std::cerr << "invalid" << std::endl;
        }
        if (!visited[nbr]) {
          cur_level->insert(nbr);
          visited[nbr] = true;
        }
      }
    }

    if (cur_level->empty()) {
      break;
    }

    std::cerr << "Level #" << level << " : " << cur_level->size() << " nodes"
              << std::endl;

    // create a new set
    tsl::robin_set<unsigned> add(cur_level->size());
    add.insert(cur_level->begin(), cur_level->end());
    bfs_order->push_back(add);

    // swap cur_level and prev_level, increment level
    prev_level->clear();
    std::swap(prev_level, cur_level);
    level++;
  }

  // cleanup
  delete cur_level;
  delete prev_level;
}

void average_degree(const VecVec &nsg, const VecSet &bfs_order) {
  unsigned level = 0;
  double   lvl_degree = 0;
  for (const auto &lvl : bfs_order) {
    lvl_degree = 0;
    for (const auto &id : lvl) {
      lvl_degree += nsg[id].size();
    }
    std::cout << "Level #" << level
              << " : Avg degree = " << lvl_degree / (double) lvl.size()
              << std::endl;
    level++;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << argv[0] << " nsg_path" << std::endl;
    exit(-1);
  }

  VecVec nsg;
  tsl::robin_map<unsigned, VecSet *> bfs_orders;

  unsigned ep_, width;
  load_nsg(argv[1], nsg, width, ep_);
  std::cout << "nsg.size() = " << nsg.size() << std::endl;
  bool *visited = new bool[nsg.size()]();
  std::fill(visited, visited + nsg.size(), false);

  unsigned start_node = ep_;
  bool     complete = false;
  while (!complete) {
    VecSet *bfs_order = new VecSet();
    std::cout << "Start node: " << start_node << std::endl;
    nsg_bfs(nsg, start_node, bfs_order, visited);
    bfs_orders.insert(std::make_pair(start_node, bfs_order));

    complete = true;
    for (unsigned idx = 0; idx < nsg.size(); idx++) {
      if (!visited[idx]) {
        complete = false;
        start_node = idx;
        break;
      }
    }
  }

  for (auto &k_v : bfs_orders) {
    std::cout << "Start node: " << k_v.first << std::endl;
    average_degree(nsg, *k_v.second);
  }

  for (auto &k_v : bfs_orders) {
    delete bfs_orders[k_v.first];
  }
  delete[] visited;

  return 0;
}
