// Copyright 2016 Tatsuya Shirakawa. All Rights Reserved.
#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>
#include <memory>
#include <random>
#include <unordered_map>

#include "covec/discrete_distribution.hpp"

namespace covec{

  struct Covec
  {
  public:

    template <class RandomGenerator>
    Covec(const std::vector<std::vector<std::size_t> >& each_counts,
	  RandomGenerator& gen,
	  const std::size_t dim = 128,
	  const double sigma = 1.0e-1,
	  const std::size_t neg_size = 1,
	  const double eta0 = 5e-3, // initial learning rate
	  const double eta1 = 1e-5 // final learning rate
	  )
      : num_entries_(), dim_(), neg_size_(), eta0_(), eta1_()
      , vs_(), cs_()
    {
      std::vector<std::shared_ptr<DiscreteDistribution> > probs;
      for(const auto& counts : each_counts){
	auto prob = std::make_shared<DiscreteDistribution>(counts.begin(), counts.end());
	probs.push_back(prob);
      }

      this->initialize(probs, gen, dim, sigma, neg_size, eta0, eta1);
    }

    template <class InputIterator, class RandomGenerator>
    Covec(const std::vector<std::pair<InputIterator, InputIterator> >& begs_and_ends,
	  RandomGenerator& gen,
	  const std::size_t dim = 128,
	  const double sigma = 1.0e-1,
	  const std::size_t neg_size = 1,
	  const double eta0 = 5e-3, // learning rate
	  const double eta1 = 1e-5 // final learning rate
	  )
      : num_entries_(), dim_(), neg_size_(), eta0_(), eta1_()
      , vs_(), cs_()
    {
      std::vector<std::shared_ptr<DiscreteDistribution> > probs;
      for(const auto&& beg_and_end : begs_and_ends){
	InputIterator beg=beg_and_end.first, end=beg_and_end.second;
	auto prob = std::make_shared<DiscreteDistribution>(beg, end);
	probs.push_back(prob);
      }

      this->initialize(probs, gen, dim, sigma, neg_size, eta0, eta1);
    }

    template <class RandomGenerator>
    Covec(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
	  RandomGenerator& gen,
	  const std::size_t dim = 128,
	  const double sigma = 1.0e-1,
	  const std::size_t neg_size = 1,
	  const double eta0 = 5e-3, // learning rate
	  const double eta1 = 1e-5 // final learning rate
	  )
      : num_entries_(), dim_(), neg_size_(), eta0_()
      , vs_(), cs_()
    { this->initialize(probs, gen, dim, sigma, neg_size, eta0, eta1); }

  public:

    template <class InputIterator, class RandomGenerator>
    void update_batch(InputIterator beg, InputIterator end, RandomGenerator& gen);

  public:

    inline const std::size_t order() const
    { return this->num_entries_.size(); }

    inline const std::vector<std::size_t>& num_entries() const
    { return this->num_entries_; }

    inline const std::vector<std::vector<std::vector<double> > >& vectors() const
    { return this->vs_; }

    inline const std::size_t dimension() const
    { return this->dim_; }

    inline const std::size_t neg_size() const
    { return this->neg_size_; }

    inline const double eta0() const
    { return this->eta0_; }

  public:

    template <class RandomGenerator>
    void initialize(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
		    RandomGenerator& gen,
		    const std::size_t dim = 128,
		    const double sigma = 1.0e-1,
		    const std::size_t neg_size = 1,
		    const double eta0 = 5e-3,
		    const double eta1 = 1e-5
		    );

  private:

    typedef enum { POSITIVE, NEGATIVE } POS_NEG;

    template <class Grad>
    void accumulate_grad(Grad& grads, const std::vector<std::size_t>& sample
			 , const POS_NEG pos_neg);

    // in case order == 2
    template <class Grad>
    void accumulate_grad_2(Grad& grads, const std::vector<std::size_t>& sample
			   , const POS_NEG pos_neg);

  private:
    std::vector<std::size_t> num_entries_; // order -> # of entries
    std::size_t dim_;
    std::size_t neg_size_;
    double eta0_;
    double eta1_;
    std::vector<std::shared_ptr<DiscreteDistribution> > probs_;
    std::vector<std::vector<std::vector<double> > > vs_; // order -> entry -> dim -> value
    std::vector<std::vector<std::size_t> > cs_; // counts of occurrencees

  }; // end of Covectorizer

  // -------------------------------------------------------------------------------

  template <class RandomGenerator>
  void Covec::initialize(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
			 RandomGenerator& gen,
			 const std::size_t dim,
			 const double sigma,
			 const std::size_t neg_size,
			 const double eta0,
			 const double eta1
			 )
  {
    this->probs_ = probs;
    this->dim_ = dim;
    this->neg_size_ = neg_size;
    this->eta0_ = eta0;
    this->eta1_ = eta0;

    this->num_entries_.clear();
    this->vs_.clear();
    this->cs_.clear();

    std::size_t order = this->probs_.size();
    this->vs_.resize(order);
    this->cs_.resize(order);

    std::normal_distribution<double> normal(0.0, sigma);
    for(std::size_t i = 0; i < order; ++i){
      const auto& prob = this->probs_[i];
      this->num_entries_.push_back(prob->probabilities().size());
      std::size_t num_entries = prob->probabilities().size();
      auto& vs = this->vs_[i];
      vs.resize(num_entries);
      for(std::size_t j = 0; j < num_entries; ++j){
	std::vector<double> v(this->dim_);
	for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
	  v[k] = normal(gen);
	}
	vs[j] = v;
      }

      this->cs_[i] = std::vector<std::size_t>(num_entries, 0);
    }
  }

  template <class Grad>
  void Covec::accumulate_grad(Grad& grads
			      , const std::vector<std::size_t>& sample
			      , Covec::POS_NEG pos_neg)
  {
    assert( sample.size() == this->order() );

    if( this->order() == 2 ){
      return accumulate_grad_2(grads, sample, pos_neg);
    }

    // count the occurences and
    // compute Hadamard product, inner_product, sigmoid of inner_product
    std::vector<double> Hadamard_product(this->dimension(), 1.0);
    for(std::size_t i = 0, I = this->order(); i < I; ++i){
      const auto j = sample[i];
      ++this->cs_[i][j];
      const auto& v = this->vs_[i][j];
      assert( v.size() == this->dimension() );
      for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
	Hadamard_product[k] *= v[k];
      }
    }
    double inner_product = std::accumulate(Hadamard_product.begin(), Hadamard_product.end(), 0.0);
    double sigmoid = 1.0 / ( 1 + std::exp(-inner_product) );
    double coeff = (pos_neg == POSITIVE? 1-sigmoid : -sigmoid);
    // compute gradients
    for(std::size_t i = 0, I = this->order(); i < I; ++i){
      const auto& j = sample[i];
      const auto& v = this->vs_[i][j];
      auto& grads_i = grads[i];
      auto itr = grads_i.find(j);
      if(itr == grads_i.end()){
	auto ret = grads_i.insert(std::make_pair(j, std::vector<double>(this->dimension(), 0.0)));
	itr = ret.first;
      }
      std::vector<double>& g = itr->second;
      for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
	if( Hadamard_product[k] == 0 ){ continue; }
	g[k] += coeff * Hadamard_product[k] / v[k];
      }
    }
  }

  template <class Grad>
  void Covec::accumulate_grad_2(Grad& grads
				, const std::vector<std::size_t>& sample
				, Covec::POS_NEG pos_neg)
  {
    assert( sample.size() == 2 && this->order() == 2 );

    // count the occurences and
    // compute Hadamard product, inner_product, sigmoid of inner_product
    const auto& v0 = this->vs_[0][sample[0]];
    const auto& v1 = this->vs_[1][sample[1]];
    const double inner_product =
      std::inner_product(v0.begin(), v0.end(), v1.begin(), 0.0);
    double sigmoid = 1.0 / ( 1 + std::exp(-inner_product) );
    double coeff = (pos_neg == POSITIVE? 1-sigmoid : -sigmoid);

    for(std::size_t i = 0; i < 2; ++i){
      const std::size_t j = sample[i];
      ++this->cs_[i][j];
      auto& grads_i = grads[i];
      auto itr = grads_i.find(j);
      if(itr == grads_i.end()){
	auto ret = grads_i.insert(std::make_pair(j, std::vector<double>(this->dimension(), 0.0)));
	itr = ret.first;
      }
      auto& g = itr->second;
      const auto& v = (i == 0 ? v1 : v0);
      for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
	g[k] += coeff * v[k];
      }
    }
  }


  template <class InputIterator, class RandomGenerator>
  void Covec::update_batch(InputIterator beg, InputIterator end, RandomGenerator& gen)
  {
    std::vector<std::unordered_map<std::size_t, std::vector<double> > > grads(this->order());

    // accumulate gradients from positive samples
    std::size_t pos_size = 0;
    for(auto itr = beg; itr != end; ++itr){
      ++pos_size;
      accumulate_grad(grads, *itr, POSITIVE);
    }

    // generate negative_sample
    // and accumulate its gradient
    std::vector<std::size_t> negative(this->order());
    for(std::size_t m = 0, M = pos_size * this->neg_size(); m < M; ++m){
      for(std::size_t i = 0, I = this->order(); i < I; ++i){
	std::size_t j = this->probs_[i]->operator()(gen);
	negative[i] = j;
	++this->cs_[i][j];
      }
      accumulate_grad(grads, negative, NEGATIVE);
    }

    // update
    for(std::size_t i = 0, I = this->order(); i < I; ++i){
      auto& vs_i = this->vs_[i];
      const auto& cs_i = this->cs_[i];
      const auto& grad_i = grads[i];
      // update sqgs, vs
      for(const auto& elem : grad_i){
	const auto j = elem.first;
	const auto& grad_ij = elem.second;
	auto& vs_ij = vs_i[j];
	for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
	  const double grad_ijk = grad_ij[k];
	  if(grad_ijk != 0){
	    const double cs_ij = cs_i[j];
	    double eta = (this->eta0_ > this->eta1_ * cs_ij ?
			  this->eta0_ / cs_ij : this->eta1_);
	    // if( this->eta0_ > this->eta1_ * cs_ij ){
	    // }
	    // double eta = std::max(this->eta0_ / cs_i[j], this->eta1_);
	    vs_ij[k] += eta * grad_ijk;
	  }
	}
      }

    }


  } // end of update_batch

} // end of namespace covec

