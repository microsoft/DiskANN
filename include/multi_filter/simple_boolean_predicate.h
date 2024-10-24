#pragma once
#include <vector>

namespace diskann {

enum BooleanOperator
{
  AND, OR
};

/// <summary>
/// Represents a simple boolean filter condition with only
/// one kind of operator. The operator can be either AND or
/// OR. The NOT operator is not supported. The predicates
/// are expected to be integers representing predicates 
/// provided by the user.
/// </summary>
/// <typeparam name="T"></typeparam>
template <typename T>
class SimpleBooleanPredicate : public AbstractPredicate
{
  public: 
    SimpleBooleanPredicate(BooleanOperator op)
    {
        _op = op;
    }
    void add_predicate(const T &predicate)
    {
        _predicates.push_back(predicate);
    }
    const std::vector<T> &get_predicates() const
    {
        return _predicates;
    }
    const BooleanOperator get_op() const
    {
        return _op;
    }

private:
    BooleanOperator _op;
    std::vector<T> _predicates;
};
}