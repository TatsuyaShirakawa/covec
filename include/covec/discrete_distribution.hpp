// Copyright 2016 Tatsuya Shirakawa. All Rights Reserved.
#pragma once

#include <vector>
#include <random>
#include <memory>
#include <cassert>

#include "covec/huffman_binary_tree.hpp"

namespace covec{

  struct DiscreteDistribution
  {
  public:
    typedef std::size_t result_type;
  private:
    typedef Node node_type;
  public:

    template <class InputIterator>
    DiscreteDistribution(InputIterator beg, InputIterator end)
      : probabilities_(), root_(nullptr)
    {
      this->set_probabilities(beg, end);
      root_ = create_Huffman_binary_tree(beg, end);
    }

    explicit DiscreteDistribution(const std::vector<double>& values)
      : probabilities_(), root_(nullptr)
    {
      this->set_probabilities(values.begin(), values.end());
      root_ = create_Huffman_binary_tree(values);
    }

  public:

    inline const std::vector<double>& probabilities() const
    { return this->probabilities_; }

    template <class Generator>
    inline std::size_t operator()( Generator& g ) const
    {
      const node_type* n = root_.get();
      while(!n->is_leaf()){
	double lv(n->left().value()), rv(n->right().value());
	double x = std::uniform_real_distribution<double>(0., lv+rv)(g);
	n = (x <= lv)? &n->left() : &n->right();
      }
      return n->key();
    }

  private:

    template <class InputIterator>
    inline void set_probabilities(InputIterator beg, InputIterator end)
    {
      probabilities_.clear();
      double Z = 0.0;
      for(InputIterator itr=beg; itr != end; ++itr){
	const double value = *itr;
	assert( value >= 0 );
	probabilities_.push_back(value);
	Z += value;
      }

      assert( Z > 0 );

      for(std::size_t i=0; i<probabilities_.size(); ++i){
	probabilities_[i] /= Z;
      }
    }

  private:
    std::vector<double> probabilities_;
    std::unique_ptr<const node_type> root_;
  }; // end of DiscreteDistribution


} // end of covec
