#include "neighbor.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "utils.h"
#include "v2/merge_insert.h"
#include <csignal>
#include <mutex>
#include <thread>
#include <vector>
#include <limits>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <numeric>
#include <omp.h>
#include <random>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include "tcmalloc/malloc_extension.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "aux_utils.h"
#include "exceptions.h"
#include "index.h"
#include "pq_flash_index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "parameters.h"
#include "partition_and_pq.h"

#include "logger.h"

#include "Neighbor_Tag.h"
#ifdef _WINDOWS
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#else
#include "linux_aligned_file_reader.h"
#endif

namespace diskann {
  template<typename T, typename TagT>
  MergeInsert<T, TagT>::MergeInsert(
      Parameters& parameters, size_t dim, const std::string mem_prefix,
      const std::string disk_prefix_in, const std::string disk_prefix_out,
      Distance<T>* dist, diskann::Metric dist_metric, bool single_file_index,
      std::string working_folder)
      : _dim(dim), _dist_metric(dist_metric),  _active_0(true), _active_1(false), _active_del_0(true),
        _active_del_1(false), _clearing_index_0(false),
        _clearing_index_1(false), _switching_disk_prefixes(false),
        _check_switch_index(false), _check_switch_delete(false) {
    _merge_th = MERGE_TH;
    _single_file_index = single_file_index;
    this->_dist_metric = dist_metric;
    _mem_index_0 = std::make_shared<diskann::Index<T, TagT>>(
        this->_dist_metric, dim, _merge_th * 2, 1, _single_file_index, 1);
    _mem_index_1 = std::make_shared<diskann::Index<T, TagT>>(
        this->_dist_metric, dim, _merge_th * 2, 1, _single_file_index, 1);

    _paras_mem.Set<unsigned>("L", parameters.Get<unsigned>("L_mem"));
    _paras_mem.Set<unsigned>("R", parameters.Get<unsigned>("R_mem"));
    _paras_mem.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_mem.Set<float>("alpha", parameters.Get<float>("alpha_mem"));
    _paras_mem.Set<unsigned>("num_rnds", 2);
    _paras_mem.Set<bool>("saturate_graph", 0);

    _paras_disk.Set<unsigned>("L", parameters.Get<unsigned>("L_disk"));
    _paras_disk.Set<unsigned>("R", parameters.Get<unsigned>("R_disk"));
    _paras_disk.Set<unsigned>("C", parameters.Get<unsigned>("C"));
    _paras_disk.Set<float>("alpha", parameters.Get<float>("alpha_disk"));
    _paras_disk.Set<unsigned>("num_rnds", 2);
    _paras_disk.Set<bool>("saturate_graph", 0);

    _num_search_threads = parameters.Get<_u32>("num_search_threads");
    _beamwidth = parameters.Get<uint64_t>("beamwidth");
    _num_nodes_to_cache = parameters.Get<_u32>("nodes_to_cache");

    _search_tpool = new ThreadPool(_num_search_threads);

    _mem_index_prefix = mem_prefix;
    _deleted_tags_file = mem_prefix + "_deleted.tags";
    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    _disk_index = new diskann::PQFlashIndex<T, TagT>(
        this->_dist_metric, reader, _single_file_index, true);

    std::string pq_prefix = _disk_index_prefix_in + "_pq";
    std::string disk_index_file = _disk_index_prefix_in + "_disk.index";
    int         res =
        _disk_index->load(_disk_index_prefix_in.c_str(), _num_search_threads);
    if (res != 0) {
      diskann::cout << "Failed to load disk index in MergeInsert constructor"
                    << std::endl;
      exit(-1);
    }

    TMP_FOLDER = working_folder;
  }

  template<typename T, typename TagT>
  MergeInsert<T, TagT>::~MergeInsert() {
    // put in destructor code
  }
 
 template<typename T, typename TagT>
 void MergeInsert<T,TagT>::construct_index_merger()
 {
     uint32_t range = _paras_disk.Get<unsigned>("R");
     uint32_t l_index = _paras_disk.Get<unsigned>("L");
     uint32_t maxc = _paras_disk.Get<unsigned>("C");
     float alpha = _paras_disk.Get<float>("alpha");
    _merger = new diskann::StreamingMerger<T, TagT>((uint32_t) _dim, _dist_comp, _dist_metric, (uint32_t) _beamwidth, range, l_index, alpha, maxc, _single_file_index);
    diskann::cout << "Created index merger object" << std::endl;

 }

 template<typename T, typename TagT>
     void MergeInsert<T, TagT>::destruct_index_merger()
     {
        delete(_merger);
        _merger = nullptr;
     }

 template<typename T, typename TagT>
 int MergeInsert<T, TagT>::insert(const T* point, const TagT& tag)
 {
     while(_check_switch_index.load())
     {
         std::this_thread::sleep_for(std::chrono::milliseconds(100));
     }
         std::shared_lock<std::shared_timed_mutex> lock(_index_lock);
         if((_active_index == 0) && (_active_0.load() == false))
         {
             diskann::cout << "Active index indicated as _mem_index_0 but it cannot accept insertions" << std::endl;
             return -1;
         }
         if((_active_index == 1) && (_active_1.load() == false))
         {
             diskann::cout << "Active index indicated as _mem_index_1 but it cannot accept insertions" << std::endl;
             return -1;
         }

         if(_active_index == 0)
         {
             if(_mem_index_0->get_num_points() < _mem_index_0->return_max_points())
             {
                 if(_mem_index_0->insert_point(point, _paras_mem, tag) != 0)
                 {
                     diskann::cout << "Could not insert point with tag " << tag << std::endl;
                     return -3;
                 }
                 {
                 std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
                 _mem_points++;
                 }
             return 0;
             }
             else
             {
                 diskann::cout << "Capacity exceeded" << std::endl;
             }
         }
         else
         {
             if(_mem_index_1->get_num_points() < _mem_index_1->return_max_points())
             {
                 if(_mem_index_1->insert_point(point, _paras_mem, tag) != 0)
                 {
                     diskann::cout << "Could not insert point with tag " << tag << std::endl;
                     return -3;
                 }
                 {
                 std::unique_lock<std::shared_timed_mutex> lock(_change_lock);
                 _mem_points++;
                 }
                return 0;
             }
             else
             {
                 diskann::cout << "Capacity exceeded in mem_index 1" << std::endl;
             }
         }
        
    return -2;
 }


 template<typename T, typename TagT>
 void MergeInsert<T,TagT>::lazy_delete(const TagT& tag)
 {
      std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
      if((_active_delete_set == 0) && (_active_del_0.load() == false))
      {
	      diskann::cout << "Active deletion set indicated as _deletion_set_0 but it cannot accept deletions" << std::endl;
      }
      if((_active_delete_set == 1) && (_active_del_1.load() == false))
      {
	      diskann::cout << "Active deletion set indicated as _deletion_set_1 but it cannot accept deletions" << std::endl;
      }

      if(_active_delete_set == 0)
	      _deletion_set_0.insert(tag);
      else
	      _deletion_set_1.insert(tag);
 }

 template<typename T, typename TagT>
    void MergeInsert<T,TagT>::search_sync(const T* query, const uint64_t K, const uint64_t search_L,
                     TagT* tags, float * distances, QueryStats * stats)
    {
        std::set<Neighbor_Tag<TagT>> best;
        //search disk index and get top K tags
        {
        std::shared_lock<std::shared_timed_mutex> lock(_disk_lock);
        assert(_switching_disk_prefixes == false);
            std::vector<uint64_t> disk_result_ids_64(search_L);
            std::vector<float> disk_result_dists(search_L);
            std::vector<TagT> disk_result_tags(search_L);
            _disk_index->cached_beam_search(
                query, search_L, search_L, disk_result_tags.data(), disk_result_dists.data(), _beamwidth,
                stats);
            for(unsigned i = 0; i < disk_result_tags.size(); i++)
            {
                Neighbor_Tag<TagT> n;
                n = Neighbor_Tag<TagT>(disk_result_tags[i], disk_result_dists[i]);
//                    best.insert(Neighbor_Tag<TagT>(disk_result_tags[i], disk_result_dists[i]));
                best.insert(n);                    
            }
        }
        //check each memory index - if non empty and not being currently cleared - search and get top K active tags 
        {
            if(_clearing_index_0.load() == false) 
            {
                std::shared_lock<std::shared_timed_mutex> lock(_clear_lock_0);
                if(_mem_index_0->get_num_points() > 0)
                {
                std::vector<Neighbor_Tag<TagT>> best_mem_index_0;
                _mem_index_0->search(query, (uint32_t)search_L, (uint32_t)search_L, best_mem_index_0);
                for(auto iter : best_mem_index_0)
                        best.insert(iter);
                }
            }
        
            if(_clearing_index_1.load() == false) 
            {
                std::shared_lock<std::shared_timed_mutex> lock(_clear_lock_1);
                if(_mem_index_1->get_num_points() > 0)
                {
                std::vector<Neighbor_Tag<TagT>> best_mem_index_1;
                _mem_index_1->search(query, (uint32_t)search_L, (uint32_t)search_L, best_mem_index_1);
                for(auto iter : best_mem_index_1)
                        best.insert(iter);
                }
            }
        }
        std::vector<Neighbor_Tag<TagT>> best_vec;
        for(auto iter : best)
            best_vec.emplace_back(iter);
//        std::sort(best_vec.begin(), best_vec.end());
        if (best_vec.size() > K)
//          best_vec.erase(best_vec.begin() + K, best_vec.end());
        //aggregate results, sort and pick top K candidates
    {
      std::shared_lock<std::shared_timed_mutex> lock(_delete_lock);
      size_t                                    pos = 0;
      for (auto iter : best_vec) {
        if((_deletion_set_0.find(iter.tag) == _deletion_set_0.end()) && (_deletion_set_1.find(iter.tag) == _deletion_set_1.end())) {
          tags[pos] = iter.tag;
          distances[pos] = iter.dist;
          pos++;
        }
        if (pos == K)
          break;
      }
    }
    }

 template<typename T, typename TagT>
 int MergeInsert<T,TagT>::trigger_merge()
 {
     if(_mem_points >= _merge_th)
     {
         save_del_set();
         switch_index();
         return 1;
     }
     return 0;
 }

 template<typename T, typename TagT>
 void MergeInsert<T,TagT>::final_merge()
 {
     diskann::cout << "Inside final_merge()." << std::endl;
     diskann::cout << _mem_index_0->get_num_points() << "  " << _mem_index_1->get_num_points() << std::endl;
     if(_mem_points > 0)
     {
         save_del_set();
         switch_index();
     }
     diskann::cout << _mem_index_0->get_num_points() << "  " << _mem_index_1->get_num_points() << std::endl;
 }

 template<typename T, typename TagT>
 void MergeInsert<T,TagT>::merge()
 {
     std::vector<std::string> mem_in;
     if(_active_index == 0)
         mem_in.push_back(_mem_index_prefix + "_1");
     else
         mem_in.push_back(_mem_index_prefix + "_0");
     std::string folder = " ";
     _merger->merge(_disk_index_prefix_in.c_str(), mem_in, _disk_index_prefix_out.c_str(), _deleted_tags_vector, folder);

    diskann::cout << "Merge done" << std::endl;
    {
    std::unique_lock<std::shared_timed_mutex> lock(_disk_lock);
            bool expected_value = false;
            if (_switching_disk_prefixes.compare_exchange_strong(expected_value, true)) {
              diskann::cout << "Switching to latest merged disk index " << std::endl;
            } else {
              diskann::cout << "Failed to switch" << std::endl;
//              return -1;
            }

    std::string temp = _disk_index_prefix_out;
    _disk_index_prefix_out = _disk_index_prefix_in;
    _disk_index_prefix_in = temp;
    delete(_disk_index);
    _disk_index = new diskann::PQFlashIndex<T,TagT>(this->_dist_metric, reader, _single_file_index, true);

    std::string pq_prefix = _disk_index_prefix_in + "_pq";
    std::string disk_index_file = _disk_index_prefix_in + "_disk.index";
    int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_search_threads);
    if(res != 0)
    {
        diskann::cout << "Failed to load new disk index after merge" << std::endl;
        exit(-1);
    }
    expected_value = true;
    _switching_disk_prefixes.compare_exchange_strong(expected_value,false);
    }
 }

 template<typename T, typename TagT>
 void MergeInsert<T,TagT>::switch_index()
 {
     //unique lock throughout the function to ensure another thread does not flip the value of _active_index after it has been saved by one thread, 
     //and multiple threads do not save the same index
     //unique lock is acquired when no other thread holds any shared lock over it, so this function will wait till any on-going insertions are completed
     //and then change the value of all related flags
     {
         bool expected_value = false;
         _check_switch_index.compare_exchange_strong(expected_value, true);
        std::unique_lock<std::shared_timed_mutex> lock(_index_lock);
        //make new index active
        if(_active_index == 0)
        {
            _mem_index_1 = std::make_shared<diskann::Index<T, TagT>>(this->_dist_metric, _dim, _merge_th * 2 , 1, _single_file_index, 1);
            bool expected_active = false;
            if (_active_1.compare_exchange_strong(expected_active, true)) {
              diskann::cout << "Initialised new index for _mem_index_1 " << std::endl;
            } else {
              diskann::cout << "Failed to initialise new _mem_index_1" << std::endl;
//              return -1;
            }
            
        }
        else
        {
            _mem_index_0 = std::make_shared<diskann::Index<T, TagT>>(this->_dist_metric, _dim, _merge_th * 2, 1, _single_file_index, 1);
            bool expected_active = false;
            if (_active_0.compare_exchange_strong(expected_active, true)) {
              diskann::cout << "Initialised new index for _mem_index_0 " << std::endl;
            } else {
              diskann::cout << "Failed to initialise new _mem_index_0" << std::endl;
  //            return -1;
            }
        }
        _active_index = 1 - _active_index;
        _mem_points = 0;
        expected_value = true;
         _check_switch_index.compare_exchange_strong(expected_value, false);

     }

     save();
     //start timer
     diskann::Timer timer;
     construct_index_merger();
     destruct_index_merger();
     diskann::cout << "Merge time : " << timer.elapsed()/1000 << " ms" << std::endl;
     //end timer

     {
         std::shared_lock<std::shared_timed_mutex> lock(_index_lock);
        //make older index inactive after merge is complete or before ?
        if(_active_index == 0)
        {
            bool expected_clearing = false;
            bool expected_active = true;
            _clearing_index_1.compare_exchange_strong(expected_clearing, true);
            {
            std::unique_lock<std::shared_timed_mutex> lock(_clear_lock_1);
            _mem_index_1.reset();
            _mem_index_1 = nullptr;
            _mem_index_1 = std::make_shared<diskann::Index<T, TagT>>(
            _dist_metric, _dim, _merge_th * 2, 1, _single_file_index, 1);
            }
            expected_clearing = true;
            assert(expected_clearing == true);
            _clearing_index_1.compare_exchange_strong(expected_clearing, false);
            assert(expected_active == true);
            _active_1.compare_exchange_strong(expected_active, false);
        }
        else
        {
            bool expected_clearing = false;
            bool expected_active = true;
            _clearing_index_0.compare_exchange_strong(expected_clearing, true);
            std::unique_lock<std::shared_timed_mutex> lock(_clear_lock_0);
            {
            _mem_index_0.reset();
            _mem_index_0 = nullptr;
            _mem_index_0 = std::make_shared<diskann::Index<T, TagT>>(
            _dist_metric, _dim, _merge_th * 2,  1, _single_file_index, 1);
            }
            expected_clearing = true;
            assert(expected_clearing == true);
            _clearing_index_0.compare_exchange_strong(expected_clearing, false);
            assert(expected_active == true);
            _active_0.compare_exchange_strong(expected_active, false);
        }
        //if merge() has returned, clear older active index
     }

 }

 template<typename T, typename TagT>
 int MergeInsert<T,TagT>::save()
 {
     //only switch_index will call this function
    bool expected_active = true;
    if(_active_index == 1)
    {
    if (_active_0.compare_exchange_strong(expected_active, false)) {
      diskann::cout << "Saving mem index 0 to merge it into disk index" << std::endl;
      std::string save_path = _mem_index_prefix + "_0";
      _mem_index_0->save(save_path.c_str());
    } else {
      diskann::cout << "Index 0 is already inactive" << std::endl;
      return -1;
    }
    }
    else
    {
    if (_active_1.compare_exchange_strong(expected_active, false)) {
      diskann::cout << "Saving mem index 1 to merge it into disk index" << std::endl;
      std::string save_path = _mem_index_prefix + "_1";
      _mem_index_1->save(save_path.c_str());
    } else {
      diskann::cout << "Index 1 is already inactive" << std::endl;
      return -1;
    }
    }
    diskann::cout << "Saved mem index" << std::endl;
    return 0;
 }

 template<typename T, typename TagT>
     void MergeInsert<T,TagT>::save_del_set()
    {
	    {
		    bool expected_value = false;
		    _check_switch_delete.compare_exchange_strong(expected_value, true);
		    std::unique_lock<std::shared_timed_mutex> lock(_delete_lock);
		    if(_active_delete_set == 0)
		    {
			     _deletion_set_1.clear();
			     bool expected_active = false;
			     if(_active_del_1.compare_exchange_strong(expected_active, true)) {
				     diskann::cout << "Cleared _deletion_set_1 - ready to accept new points" << std::endl;
			     }
			     else
			     {
				     diskann::cout << "Failed to clear _deletion_set_1" << std::endl;
			   	}
		     }
		     else
		     {
			     _deletion_set_0.clear();
			     bool expected_active = false;
			     if(_active_del_0.compare_exchange_strong(expected_active, true)) {
				     diskann::cout << "Cleared _deletion_set_0 - ready to accept new points" << std::endl;
			     }
			     else
			     {
				     diskann::cout << "Failed to clear _deletion_set_0" << std::endl;
			     }
		     }
		     _active_delete_set = 1 - _active_delete_set;
		     bool expected_active  = true;
		     if(_active_delete_set == 0)
			     _active_del_1.compare_exchange_strong(expected_active, false);
		     else
			     _active_del_0.compare_exchange_strong(expected_active, false);
		     expected_value = true;
		     _check_switch_delete.compare_exchange_strong(expected_value, false);
	     }

	 if(_active_delete_set == 0)
	 {
         std::vector<TagT> * del_vec = new std::vector<TagT>(_deletion_set_1.size());
         
		 size_t i = 0;
		 for(auto iter : _deletion_set_1)
         	{
                (*del_vec)[i] = iter;
             		i++;
         	}
         _deleted_tags_vector.clear();
         _deleted_tags_vector.push_back(del_vec);
	 }
	 else
	 {
         std::vector<TagT> * del_vec = new std::vector<TagT>(_deletion_set_0.size());
         
		 size_t i = 0;
		 for(auto iter : _deletion_set_0)
         	{
                (*del_vec)[i] = iter;
             		i++;
         	}
         _deleted_tags_vector.clear();
         _deleted_tags_vector.push_back(del_vec);
	 }
		 
     }
    
 template<typename T, typename TagT>
    std::string MergeInsert<T,TagT>::ret_merge_prefix()
 {
     return _disk_index_prefix_in;
 }
  // template class instantiations
  template class MergeInsert<float, uint32_t>;
  template class MergeInsert<uint8_t, uint32_t>;
  template class MergeInsert<int8_t, uint32_t>;
  template class MergeInsert<float, int64_t>;
  template class MergeInsert<uint8_t, int64_t>;
  template class MergeInsert<int8_t, int64_t>;
  template class MergeInsert<float, uint64_t>;
  template class MergeInsert<uint8_t, uint64_t>;
  template class MergeInsert<int8_t, uint64_t>;
}
