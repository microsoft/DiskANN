#pragma once
#include "multi_filter/abstract_predicate.h"
#include "types.h"
#include <vector>
namespace diskann {
template <typename LabelT> class AbstractFilterStore {
public:
  /// <summary>
  /// Returns the filters for a data point. Only valid for base points
  /// </summary>
  /// <param name="point">base point id</param>
  /// <returns>list of filters of the base point</returns>
  virtual const std::vector<LabelT> &
  get_filters_for_point(location_t point) const = 0;

  /// <summary>
  /// Adds filters for a point.
  /// </summary>
  /// <param name="point"></param>
  /// <param name="filters"></param>
  virtual void add_filters_for_point(location_t point,
                                     const std::vector<LabelT> &filters) = 0;

  /// <summary>
  /// Returns a score between [0,1] indicating how many points in the dataset
  /// matched the predicate
  /// </summary>
  /// <param name="pred">Predicate to match</param>
  /// <returns>Score between [0,1] indicate %age of points matching
  /// pred</returns>
  virtual float
  get_predicate_selectivity(const AbstractPredicate &pred) const = 0;
};

} // namespace diskann
