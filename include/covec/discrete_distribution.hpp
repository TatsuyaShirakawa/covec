// Copyright 2016 Tatsuya Shirakawa. All Rights Reserved.
#pragma once

#include <vector>
#include <random>
#include <memory>

#include "covec/huffman_binary_tree.hpp"

namespace covec{

  struct DiscreteDistribution
  {
  public:
    typedef std::size_t result_type;
  private:
    typedef Node<double> node_type;
  public:
    explicit DiscreteDistribution(const std::vector<double>& values)
      : root_(nullptr)
    { root_ = create_Huffman_binary_tree(values); }

  public:

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
    std::unique_ptr<const node_type> root_;
  }; // end of DiscreteDistribution


} // end of covec



  
