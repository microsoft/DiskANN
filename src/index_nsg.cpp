#include <efanna2e/index_nsg.h>
#include <efanna2e/exceptions.h>
#include <efanna2e/parameters.h>
#include <omp.h>
#include <chrono>
#include <boost/dynamic_bitset.hpp>
#include <bitset>
#include <cmath>

namespace efanna2e {
#define _CONTROL_NUM 100
IndexNSG::IndexNSG(const size_t dimension, const size_t n, Metric m, Index *initializer) : Index(dimension, n, m),
                                                                                           initializer_{initializer} {
}

IndexNSG::~IndexNSG() {}

void IndexNSG::Save(const char *filename) {
  std::ofstream out(filename, std::ios::binary | std::ios::out);
  assert(final_graph_.size() == nd_);

  out.write((char *) &width, sizeof(unsigned));
  out.write((char *) &ep_, sizeof(unsigned));
  for (unsigned i = 0; i < nd_; i++) {
    unsigned GK = (unsigned) final_graph_[i].size();
    out.write((char *) &GK, sizeof(unsigned));
    out.write((char *) final_graph_[i].data(), GK * sizeof(unsigned));
  }
  out.close();
}

void IndexNSG::Load(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  in.read((char *) &width, sizeof(unsigned));
  in.read((char *) &ep_, sizeof(unsigned));
  //width=100;
  unsigned cc=0;
  while (!in.eof()) {
    unsigned k;
    in.read((char *) &k, sizeof(unsigned));
    if (in.eof())break;
    cc += k;
    std::vector<unsigned> tmp(k);
    in.read((char *) tmp.data(), k * sizeof(unsigned));
    final_graph_.push_back(tmp);
  }
  cc /= nd_;
  std::cout<<cc<<std::endl;
}
void IndexNSG::Load_nn_graph(const char *filename) {
  std::ifstream in(filename, std::ios::binary);
  unsigned k;
  in.read((char *) &k, sizeof(unsigned));
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  size_t fsize = (size_t) ss;
  size_t num = (unsigned) (fsize / (k + 1) / 4);
  in.seekg(0, std::ios::beg);

  final_graph_.resize(num);
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    final_graph_[i].resize(k);
    in.read((char *) final_graph_[i].data(), k * sizeof(unsigned));
  }
  in.close();
}

void IndexNSG::get_neighbors(
    const float *query,
    const Parameters &parameter,
    std::vector <Neighbor> &retset, std::vector <Neighbor> &fullset) {
  unsigned L = parameter.Get<unsigned>("L");

  retset.resize(L + 1);
  std::vector<unsigned> init_ids(L);
  //initializer_->Search(query, nullptr, L, parameter, init_ids.data());

  boost::dynamic_bitset<> flags{nd_, 0};
  L = 0;
  for(unsigned i=0; i < init_ids.size() && i < final_graph_[ep_].size(); i++){
    init_ids[i] = final_graph_[ep_][i];
    flags[init_ids[i]] = true;
    L++;
  }
  while(L < init_ids.size()){
    unsigned id = rand() % nd_;
    if(flags[id])continue;
    init_ids[L] = id;
    L++;
    flags[id] = true;
  }

  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if(id >= nd_)continue;
    //std::cout<<id<<std::endl;
    float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned) dimension_);
    retset[i] = Neighbor(id, dist, true);
    //flags[id] = 1;
    L++;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int) L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id])continue;
        flags[id] = 1;

        float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned) dimension_);
	      Neighbor nn(id, dist, true);
	      fullset.push_back(nn);
        if (dist >= retset[L - 1].distance)continue;
        int r = InsertIntoPool(retset.data(), L, nn);

        if(L+1 < retset.size()) ++L;
        if (r < nk)nk = r;
      }

    }
    if (nk <= k)k = nk;
    else ++k;
  }
}

void IndexNSG::init_graph(const Parameters &parameters) {
  float *center = new float[dimension_];
  for (unsigned j = 0; j < dimension_; j++)center[j] = 0;
  for (unsigned i = 0; i < nd_; i++) {
    for (unsigned j = 0; j < dimension_; j++) {
      center[j] += data_[i * dimension_ + j];
    }
  }
  for (unsigned j = 0; j < dimension_; j++) {
    center[j] /= nd_;
  }
  std::vector <Neighbor> tmp, pool;
  get_neighbors(center, parameters, tmp, pool);
  ep_ = tmp[0].id;
}

void IndexNSG::add_cnn(unsigned des, Neighbor p, unsigned range, LockGraph &cut_graph_) {
  LockGuard guard(cut_graph_[des].lock);
  for (unsigned i = 0; i < cut_graph_[des].pool.size(); i++) {
    if (p.id == cut_graph_[des].pool[i].id)return;
  }
  cut_graph_[des].pool.push_back(p);
  if (cut_graph_[des].pool.size() > range) {
    std::vector <Neighbor> result;
    std::vector <Neighbor> &pool = cut_graph_[des].pool;
    unsigned start = 0;
    std::sort(pool.begin(), pool.end());
    result.push_back(pool[start]);

    while (result.size() < range && (++start) < pool.size()) {
      auto &p = pool[start];
      bool occlude = false;
      for (unsigned t = 0; t < result.size(); t++) {
        if (p.id == result[t].id) {
          occlude = true;
          break;
        }
        float djk = distance_->compare(data_ + dimension_ * result[t].id, data_ + dimension_ * p.id, dimension_);
        if (djk < p.distance/* dik */) {
          occlude = true;
          break;
        }

      }
      if (!occlude)result.push_back(p);
    }
    pool.swap(result);
  }

}
void IndexNSG::sync_prune(unsigned q,
                          std::vector <Neighbor> &pool,
                          const Parameters &parameter,
                          LockGraph &cut_graph_) {
  unsigned range = parameter.Get<unsigned>("R");
  width = range;
  unsigned start = 0;

  boost::dynamic_bitset<> flags{nd_, 0};
  for (unsigned i = 0; i < pool.size(); i++)flags[pool[i].id] = 1;
  for (unsigned nn = 0; nn < final_graph_[q].size(); nn++) {
    unsigned id = final_graph_[q][nn];
    if (flags[id])continue;
    float dist = distance_->compare(data_ + dimension_ * q, data_ + dimension_ * id, dimension_);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());
  std::vector <Neighbor> result;
  if(pool[start].id == q)start++;
  result.push_back(pool[start]);

  while (result.size() < range && (++start) < pool.size()) {
    auto &p = pool[start];
    bool occlude = false;
    for (unsigned t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = distance_->compare(data_ + dimension_ * result[t].id, data_ + dimension_ * p.id, dimension_);
      if (djk < p.distance/* dik */) {
        occlude = true;
        break;
      }

    }
    if (!occlude)result.push_back(p);
  }
  for (unsigned t = 0; t < result.size(); t++) {
    add_cnn(q, result[t], range, cut_graph_);
    add_cnn(result[t].id, Neighbor(q, result[t].distance, true), range, cut_graph_);
  }
}

void IndexNSG::Link(const Parameters &parameters, LockGraph &cut_graph_) {
  std::cout << " graph link" << std::endl;
  unsigned progress=0;
  unsigned percent = 10;
  unsigned step_size = nd_/percent;
  std::mutex progress_lock;

#pragma omp parallel
{
  unsigned cnt = 0;
#pragma omp for
  for (unsigned n = 0; n < nd_; ++n) {
    std::vector <Neighbor> pool, tmp;
    get_neighbors(data_ + dimension_ * n, parameters, tmp, pool);
    sync_prune(n, pool, parameters, cut_graph_);

    cnt++;
    if(cnt % step_size == 0){
      LockGuard g(progress_lock);
      std::cout<<progress++ <<"/"<< percent << " completed" <<std::endl;
    }
  }
}

}


void IndexNSG::Build(size_t n, const float *data, const Parameters &parameters) {
  std::string nn_graph_path = parameters.Get<std::string>("nn_graph_path");
  Load_nn_graph(nn_graph_path.c_str());
  data_ = data;
  init_graph(parameters);
  LockGraph cut_graph_(nd_);
  Link(parameters, cut_graph_);
  final_graph_.resize(nd_);
  unsigned max = 0, min = 1e6, avg = 0, cnt=0;
  for (unsigned i = 0; i < nd_; i++) {
    auto &pool = cut_graph_[i].pool;
    max = max < pool.size() ? pool.size() : max;
    min = min > pool.size() ? pool.size() : min;
    avg += pool.size();
    if(pool.size() < 2)cnt++;
    final_graph_[i].resize(pool.size());
    for (unsigned j = 0; j < pool.size(); j++) {
      final_graph_[i][j] = pool[j].id;
    }
  }
  avg /= nd_;
  std::cout << max << ":" << avg << ":" << min << ":" << cnt << "\n";
  tree_grow(parameters);
  has_built = true;
}

void IndexNSG::Search(
    const float *query,
    const float *x,
    size_t K,
    const Parameters &parameters,
    unsigned *indices) {
  const unsigned L = parameters.Get<unsigned>("L_search");
  data_ = x;
  std::vector <Neighbor> retset(L + 1);
  std::vector<unsigned> init_ids(L);
  boost::dynamic_bitset<> flags{nd_, 0};
  //std::mt19937 rng(rand());
  //GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  unsigned tmp_l = 0;
  for(; tmp_l<L && tmp_l<final_graph_[ep_].size(); tmp_l++){
    init_ids[tmp_l] = final_graph_[ep_][tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while(tmp_l < L){
    unsigned id = rand() % nd_;
    if(flags[id])continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }


  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    float dist = distance_->compare(data_ + dimension_ * id, query, (unsigned) dimension_);
    retset[i] = Neighbor(id, dist, true);
    //flags[id] = true;
  }

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int) L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      for (unsigned m = 0; m < final_graph_[n].size(); ++m) {
        unsigned id = final_graph_[n][m];
        if (flags[id])continue;
        flags[id] = 1;
        float dist = distance_->compare(query, data_ + dimension_ * id, (unsigned) dimension_);
        if (dist >= retset[L - 1].distance)continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        if (r < nk)nk = r;
      }
    }
    if (nk <= k)k = nk;
    else ++k;
  }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}

void IndexNSG::SearchWithOptGraph(
    const float *query,
    size_t K,
    const Parameters &parameters,
    unsigned *indices){
  unsigned L = parameters.Get<unsigned>("L_search");
  unsigned P = parameters.Get<unsigned>("P_search");
  DistanceFastL2* dist_fast = (DistanceFastL2*)distance_;

  P = P > K ? P : K;
  std::vector <Neighbor> retset(P + 1);
  std::vector<unsigned> init_ids(L);
  //std::mt19937 rng(rand());
  //GenRandom(rng, init_ids.data(), L, (unsigned) nd_);

  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned tmp_l = 0;
  unsigned *neighbors = (unsigned*)(opt_graph_ + node_size * ep_ + data_len);
  unsigned MaxM_ep = *neighbors;
  neighbors++;

  for(; tmp_l < L && tmp_l < MaxM_ep; tmp_l++){
    init_ids[tmp_l] = neighbors[tmp_l];
    flags[init_ids[tmp_l]] = true;
  }

  while(tmp_l < L){
    unsigned id = rand() % nd_;
    if(flags[id])continue;
    flags[id] = true;
    init_ids[tmp_l] = id;
    tmp_l++;
  }

  for (unsigned i = 0; i < init_ids.size(); i++){
    unsigned id = init_ids[i];
    if(id >= nd_)continue;
    _mm_prefetch(opt_graph_ + node_size * id, _MM_HINT_T0);
  }
  L = 0;
  for (unsigned i = 0; i < init_ids.size(); i++) {
    unsigned id = init_ids[i];
    if(id >= nd_)continue;
    float *x = (float*)(opt_graph_ + node_size * id);
    float norm_x = *x;x++;
    float dist = dist_fast->compare(x, query, norm_x, (unsigned) dimension_);
    retset[i] = Neighbor(id, dist, true);
    flags[id] = true;
    L++;
  }
  //std::cout<<L<<std::endl;

  std::sort(retset.begin(), retset.begin() + L);
  int k = 0;
  while (k < (int) L) {
    int nk = L;

    if (retset[k].flag) {
      retset[k].flag = false;
      unsigned n = retset[k].id;

      _mm_prefetch(opt_graph_ + node_size * n + data_len, _MM_HINT_T0);
      unsigned *neighbors = (unsigned*)(opt_graph_ + node_size * n + data_len);
      unsigned MaxM = *neighbors;
      neighbors++;
      for(unsigned m=0; m<MaxM; ++m)
        _mm_prefetch(opt_graph_ + node_size * neighbors[m], _MM_HINT_T0);
      for (unsigned m = 0; m < MaxM; ++m) {
        unsigned id = neighbors[m];
        if (flags[id])continue;
        flags[id] = 1;
        float *data = (float*)(opt_graph_ + node_size * id);
        float norm = *data;data++;
        float dist = dist_fast->compare(query, data, norm, (unsigned) dimension_);
        if (dist >= retset[L - 1].distance)continue;
        Neighbor nn(id, dist, true);
        int r = InsertIntoPool(retset.data(), L, nn);

        //if(L+1 < retset.size()) ++L;
        if (r < nk)nk = r;
      }

    }
    if (nk <= k)k = nk;
    else ++k;
    }
  for (size_t i = 0; i < K; i++) {
    indices[i] = retset[i].id;
  }
}


void IndexNSG::OptimizeGraph(float* data){//use after build or load

  data_ = data;
  data_len = (dimension_ + 1) * sizeof(float);
  neighbor_len = (width + 1) * sizeof(unsigned);
  node_size = data_len + neighbor_len;
  opt_graph_ = (char*)malloc(node_size * nd_);
  DistanceFastL2* dist_fast = (DistanceFastL2*)distance_;
  for(unsigned i=0; i<nd_; i++){
    char* cur_node_offset = opt_graph_ + i * node_size;
    float cur_norm = dist_fast->norm(data_ + i * dimension_, dimension_);
    std::memcpy(cur_node_offset, &cur_norm, sizeof(float));
    std::memcpy(cur_node_offset + sizeof(float), data_ + i * dimension_, data_len-sizeof(float));

    cur_node_offset += data_len;
    unsigned k = final_graph_[i].size();
    std::memcpy(cur_node_offset, &k, sizeof(unsigned));
    std::memcpy(cur_node_offset + sizeof(unsigned), final_graph_[i].data(), k * sizeof(unsigned));
    std::vector<unsigned>().swap(final_graph_[i]);
  }
  free(data);
  data_ = nullptr;
  CompactGraph().swap(final_graph_);
}

void IndexNSG::DFS(boost::dynamic_bitset<> &flag, unsigned root, unsigned &cnt){
  unsigned tmp = root;
  std::stack<unsigned> s;
  s.push(root);
  if(!flag[root])cnt++;
  flag[root] = true;
  while(!s.empty()){

    unsigned next = nd_ + 1;
    for(unsigned i=0; i<final_graph_[tmp].size(); i++){
      if(flag[final_graph_[tmp][i]] == false){
        next = final_graph_[tmp][i];
        break;
      }
    }
    //std::cout << next <<":"<<cnt <<":"<<tmp <<":"<<s.size()<< '\n';
    if(next == (nd_ + 1)){
      s.pop();
      if(s.empty())break;
      tmp = s.top();
      continue;
    }
    tmp = next;
    flag[tmp] = true;s.push(tmp);cnt++;
  }
}

void IndexNSG::findroot(boost::dynamic_bitset<> &flag, unsigned &root, const Parameters &parameter){
  unsigned id;
  for(unsigned i=0; i<nd_; i++){
    if(flag[i] == false){
      id = i;
      break;
    }
  }
  std::vector <Neighbor> tmp, pool;
  get_neighbors(data_ + dimension_ * id, parameter, tmp, pool);
  std::sort(pool.begin(), pool.end());

  unsigned found = 0;
  for(unsigned i=0; i<pool.size(); i++){
    if(flag[pool[i].id]){
      //std::cout << pool[i].id << '\n';
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if(found == 0){
    while(true){
      unsigned rid = rand() % nd_;
      if(flag[rid]){
        root = rid;
        //std::cout << root << '\n';
        break;
      }
    }
  }
  final_graph_[root].push_back(id);

}
void IndexNSG::tree_grow(const Parameters &parameter){
  unsigned root = ep_;
  boost::dynamic_bitset<> flags{nd_, 0};
  unsigned unlinked_cnt = 0;
  while(unlinked_cnt < nd_){
    DFS(flags, root, unlinked_cnt);
    //std::cout << unlinked_cnt << '\n';
    if(unlinked_cnt >= nd_)break;
    findroot(flags, root, parameter);
    //std::cout << "new root"<<":"<<root << '\n';
  }
}

}
