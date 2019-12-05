#include "utils.h"
#include  <set>

namespace diskann {
  float calc_recall_set(unsigned num_queries, unsigned *gold_std,
                        unsigned dim_gs, unsigned *our_results, unsigned dim_or,
                        unsigned recall_at, unsigned subset_size) {
    unsigned           total_recall = 0;
    std::set<unsigned> gt, res;

    for (size_t i = 0; i < num_queries; i++) {
      gt.clear();
      res.clear();
      unsigned *gt_vec = gold_std + dim_gs * i;
      unsigned *res_vec = our_results + dim_or * i;
      gt.insert(gt_vec, gt_vec + recall_at);
      res.insert(res_vec, res_vec + subset_size);
      unsigned cur_recall = 0;
      for (auto &v : gt) {
        if (res.find(v) != res.end()) {
          cur_recall++;
        }
      }
      // std::cout << " idx: " << i << ", interesection: " << cur_recall <<
      // "\n";
      total_recall += cur_recall;
    }
    return ((float) total_recall) / ((float) num_queries) *
           (100.0 / ((float) recall_at));
  }
};
