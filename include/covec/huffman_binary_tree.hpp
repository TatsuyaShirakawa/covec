// Copyright 2016 Tatsuya Shirakawa. All Rights Reserved.
#pragma once

#include <cassert>

#include <memory>
#include <vector>


namespace covec{

  template <class V=double>
  struct Node
  {
  public:
    typedef std::size_t key_type;
    typedef V value_type;
  private:
    typedef Node<value_type> self_type;
  public:
    Node(): left_(nullptr), right_(nullptr), key_(nullptr), value_(nullptr)
    {}
	
    Node(const key_type& key, const value_type& value)
      : left_(nullptr), right_(nullptr)
      , key_(new key_type(key)), value_(new value_type(value))
    {}

    Node(std::unique_ptr<const self_type>& left,
	 std::unique_ptr<const self_type>& right)
      : left_(std::move(left)), right_(std::move(right))
      , key_(nullptr), value_(new value_type(this->left_->value() + this->right_->value()))
    {}

  public:

    // elemental APIs
    inline bool is_leaf() const { return !left_ && !right_ && key_ && value_; }
    inline const Node& left() const { return *left_; }
    inline const Node& right() const { return *right_; }
    inline const key_type& key() const { return *key_; }
    inline const value_type& value() const { return *value_; }
      
  private:
    std::unique_ptr<const Node> left_;
    std::unique_ptr<const Node> right_;
    std::unique_ptr<const key_type> key_;
    std::unique_ptr<const value_type> value_;

  }; // end of Node

  /**
   * Create Huffman binary tree from values
   */
  template <class V=double>
  std::unique_ptr<const Node<V> >
  create_Huffman_binary_tree(const std::vector<V>& values)
  {
    typedef V value_type;
    typedef Node<V> node_type;
    
    assert( !values.empty() );

    // [Huffman Tree]
    // 1. create nodes from thetas
    // 2. sort nodes by descending order of values
    // 3. until nodes.size() > 1:
    //   3-1. merge last two nodes into anew node
    //   3-2. sort nodes by descending order of values
    // 4. root = nodes[0]

    // 1. create nodes from thetas
    std::vector<std::unique_ptr<const node_type> > nodes;
    value_type Z = 0;
    for(std::size_t i=0; i < values.size(); ++i){
      assert( values[i] >= 0 );
      nodes.push_back(std::make_unique<const node_type>(i, values[i]));
      Z += values[i];
    }
    assert( Z > 0 );

    // 2. sort nodes by descending order of values
    std::sort(nodes.begin(), nodes.end()
	      , [](const std::unique_ptr<const node_type>& n1,
		   const std::unique_ptr<const node_type>& n2)
	      { return n1->value() > n2->value(); });

    // 3. until nodes.size() > 1
    while(nodes.size() > 1){
      // merge last two nodes into a new node	
      std::size_t N = nodes.size();
      auto n1 = std::move(nodes.back()); nodes.pop_back();
      auto n2 = std::move(nodes.back()); nodes.pop_back();
      auto n = std::make_unique<const node_type>(n1, n2);
      nodes.push_back(std::move(n));
      // sort nodes by descending order of values
      std::size_t n_pos = N-2;
      auto value = nodes.back()->value();
      while( n_pos > 0 && value > nodes[n_pos-1]->value() ){
	nodes[n_pos].swap(nodes[n_pos-1]);
	//	std::swap(nodes[n_pos], nodes[n_pos-1]);
	--n_pos;
      }
    }

    return std::move(nodes[0]);
  } // end of create_Huffman_binary_tree
    

} // end of covec














