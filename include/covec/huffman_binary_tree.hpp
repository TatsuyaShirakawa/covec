// Copyright 2016 Tatsuya Shirakawa. All Rights Reserved.
#pragma once

#include <cassert>

#include <memory>
#include <vector>
#include <queue>


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

    Node() noexcept: left_(nullptr), right_(nullptr), key_(nullptr), value_(0)
    {}

    Node(const key_type& key, const value_type& value) noexcept
      : left_(nullptr), right_(nullptr)
      , key_(new key_type(key)), value_(value)
    {}

    Node(const Node* left, const Node* right)
      : left_(left), right_(right)
      , key_(nullptr), value_(left->value_ + right->value_)
    {}

  public:

    // elemental APIs
    inline bool is_leaf() const { return key_ != nullptr; }
    inline const Node& left() const { return *left_; }
    inline const Node& right() const { return *right_; }
    inline const key_type& key() const { return *key_; }
    inline const value_type& value() const { return value_; }

  private:
    std::unique_ptr<const Node> left_;
    std::unique_ptr<const Node> right_;
    std::unique_ptr<const key_type> key_;
    value_type value_;

  }; // end of Node

  template <class node_ptr>
  struct node_ptr_compare
  {
    // ascending order
    inline bool operator()(node_ptr n1, node_ptr n2) const
    { return n1->value() > n2->value(); }
  };

  /**
   * Create Huffman binary tree from values
   */
  template <class V=double, class InputIterator>
  std::unique_ptr<const Node<V> >
  create_Huffman_binary_tree(InputIterator beg, InputIterator end)
  {

    typedef V value_type;
    typedef Node<V> node_type;

    assert( beg != end );

    // 1. create nodes from thetas
    std::vector<const node_type*> nodes;
    value_type Z = 0;
    std::size_t i=0;
    for(InputIterator itr=beg; itr!=end; ++itr){
      value_type value = *itr;
      nodes.push_back(new node_type(i, value));
      Z += value;
      ++i;
    }
    assert( Z > 0 );

    std::priority_queue<const node_type*, std::vector<const node_type*>, node_ptr_compare<const node_type*> >
      pq(nodes.begin(), nodes.end());

    // 3. until nodes.size() > 1
    std::size_t N=nodes.size();
    while(N > 1){
      auto n1 = pq.top(); pq.pop();
      auto n2 = pq.top(); pq.pop();
      assert( n1->value() <= n2->value() );
      pq.push(new node_type(n1, n2));
      --N;
    }
    return std::unique_ptr<const node_type>(pq.top());

  } // end of create_Huffman_binary_tree

  /**
   * Create Huffman binary tree from values
   */
  template <class V=double>
  std::unique_ptr<const Node<V> >
  create_Huffman_binary_tree(const std::vector<V>& values)
  {
    return
      create_Huffman_binary_tree<double, typename std::vector<V>::const_iterator>
      (values.cbegin(), values.cend());
  }

} // end of covec
