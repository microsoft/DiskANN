#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

typedef std::vector<std::vector<unsigned>>    VecVec;
typedef std::vector<tsl::robin_set<unsigned>> VecSet;

typedef tsl::robin_map<unsigned, unsigned> MapCount;
typedef std::vector<MapCount> VecMapCount;

void load_nsg(const char *filename, VecVec &graph, unsigned &width,
              unsigned &ep_, const bool check_for_dup = true) {
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
    in.read((char *) tmp.data(), k * sizeof(unsigned));

    if (check_for_dup) {
      tsl::robin_set<unsigned> tmp_set;
      for (unsigned id : tmp)
        tmp_set.insert(id);
      n_dups += (tmp.size() - tmp_set.size());
    }
    graph.push_back(tmp);

    if ((graph.size() % 1000000) == 0)
      std::cout << graph.size() << " nodes read" << std::endl;
  }
  std::cout << "Total # of dups: " << n_dups
            << ", avg # of dups: " << (double) n_dups / (double) graph.size()
            << std::endl;
  std::cout << "Total #nodes = " << graph.size() << std::endl;
}

// Do BFS and count in-degree for each node from previous level
void nsg_bfs(const VecVec &nsg, const unsigned start_node,
             VecMapCount &bfs_order, bool *visited) {
  auto cur_level = new MapCount();
  auto prev_level = new MapCount();
  prev_level->insert(std::make_pair(start_node, 0));
  visited[start_node] = true;
  unsigned level = 0;
  unsigned nsg_size = nsg.size();
  while (true) {
    // clear state
    cur_level->clear();

    // select candidates
    for (auto id : *prev_level) {
      for (const auto &nbr : nsg[id.first]) {
        if (nbr >= nsg_size) {
          std::cerr << "invalid" << std::endl;
        }

        if (!visited[nbr]) {
          visited[nbr] = true;
          cur_level->insert(std::make_pair(nbr, 1));
        } else {
          if (cur_level->find(nbr) != cur_level->end())
            (*cur_level)[nbr] = 1 + (*cur_level)[nbr];
        }
      }
    }

    if (cur_level->empty())
      break;

    std::cout << "Level #" << level << " : " << cur_level->size() << " nodes"
              << std::endl;

    // create a new set
    MapCount add(cur_level->size());
    add.insert(cur_level->begin(), cur_level->end());
    bfs_order.push_back(add);

    // swap cur_level and prev_level, increment level
    prev_level->clear();
    std::swap(prev_level, cur_level);
    level++;
  }

  // cleanup
  delete cur_level;
  delete prev_level;
}

void average_out_degree(const VecVec &nsg, const VecMapCount &bfs_order) {
  unsigned level = 0;
  double   lvl_degree = 0;
  for (const auto &lvl : bfs_order) {
    lvl_degree = 0;
    for (const auto &id : lvl) {
      lvl_degree += nsg[id.first].size();
    }
    std::cout << "Level #" << level
              << " : Avg degree = " << lvl_degree / (double) lvl.size()
              << std::endl;
    level++;
  }
}

void average_in_bfs_degree(const VecMapCount &bfs_order) {
  unsigned level = 0;
  double   lvl_degree = 0;
  for (const auto &lvl : bfs_order) {
    lvl_degree = 0;
    std::vector<unsigned> in_degrees;
    for (const auto &id : lvl) {
      lvl_degree += id.second;
      in_degrees.push_back(id.second);
    }
    std::sort(in_degrees.begin(), in_degrees.end(), std::greater<unsigned>());
    unsigned univalent =
        in_degrees.end() - std::find(in_degrees.begin(), in_degrees.end(), 1);
    std::cout << "Level #" << level
              << " : Avg BFS In degree = " << lvl_degree / (double) lvl.size()
              << "\tmax = " << in_degrees[0]
              << "\t99pc = " << in_degrees[0.01 * in_degrees.size()]
              << "\t95pc = " << in_degrees[0.05 * in_degrees.size()]
              << "\t90pc = " << in_degrees[0.1 * in_degrees.size()]
              << "\t50pc = " << in_degrees[0.5 * in_degrees.size()]
              << "\t#ones = " << univalent << std::endl;
    level++;
  }
}

void average_in_degree(const VecVec &nsg, const VecMapCount &bfs_order,
                       std::vector<unsigned> &in_count) {
  unsigned level = 0;
  double   lvl_degree = 0;

  assert(in_count.size() == nsg.size());
  for (auto iter = in_count.begin(); iter != in_count.end(); ++iter)
    *iter = 0;

  for (const auto &lvl : bfs_order)
    for (const auto &id : lvl)
      for (const auto ngh : nsg[id.first])
        in_count[ngh]++;

  for (const auto &lvl : bfs_order) {
    lvl_degree = 0;
    std::vector<unsigned> in_degrees;
    for (const auto &id : lvl) {
      lvl_degree += in_count[id.first];
      in_degrees.push_back(in_count[id.first]);
    }
    std::sort(in_degrees.begin(), in_degrees.end(), std::greater<unsigned>());
    unsigned univalent =
        in_degrees.end() - std::find(in_degrees.begin(), in_degrees.end(), 1);
    std::cout << "Level #" << level
              << " : Avg In degree = " << lvl_degree / (double) lvl.size()
              << "\tmax = " << in_degrees[0]
              << "\t99pc = " << in_degrees[0.01 * in_degrees.size()]
              << "\t95pc = " << in_degrees[0.05 * in_degrees.size()]
              << "\t90pc = " << in_degrees[0.1 * in_degrees.size()]
              << "\t50pc = " << in_degrees[0.5 * in_degrees.size()]
              << "\t#ones = " << univalent << std::endl;
    level++;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << argv[0] << " nsg_path" << std::endl;
    exit(-1);
  }

  VecVec nsg;

  unsigned ep_, width;
  load_nsg(argv[1], nsg, width, ep_, false);
  std::cout << "nsg.size() = " << nsg.size() << std::endl;
  bool *visited = new bool[nsg.size()]();
  std::fill(visited, visited + nsg.size(), false);

  unsigned              start_node = ep_;
  unsigned              previous_start = 0;
  bool                  complete = false;
  std::vector<unsigned> in_count(nsg.size(), 0);

  while (!complete) {
    VecMapCount bfs_order;
    std::cout << "Start node: " << start_node << std::endl;
    nsg_bfs(nsg, start_node, bfs_order, visited);

    average_out_degree(nsg, bfs_order);
    average_in_bfs_degree(bfs_order);
    average_in_degree(nsg, bfs_order, in_count);

    complete = true;
    for (unsigned idx = previous_start; idx < nsg.size(); idx++) {
      if (!visited[idx]) {
        complete = false;
        start_node = idx;
        break;
      }
    }
    previous_start = start_node + 1;
  }

  delete[] visited;

  return 0;
}
