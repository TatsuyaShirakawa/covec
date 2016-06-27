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
    
    Covec(const std::vector<std::vector<std::size_t> >& each_counts,
	  std::random_device& rd,
	  const std::size_t dim=128,
	  const double sigma=1.0e-1,	  
	  const std::size_t neg_size=1,
	  const double eta0 = 5e-3 // learning rate
	  )
      : num_entries_(), dim_(), neg_size_(), eta0_()
      , vs_(), sqgs_()
    {
      std::vector<std::shared_ptr<DiscreteDistribution> > probs;
      for(const auto& counts : each_counts){
	auto prob = std::make_shared<DiscreteDistribution>(counts.begin(), counts.end());
	probs.push_back(prob);
      }

      this->initialize(probs, rd, dim, sigma, neg_size, eta0);
    }

    template <class InputIterator>
    Covec(const std::vector<std::pair<InputIterator, InputIterator> >& begs_and_ends,
	  std::random_device& rd,	  
	  const std::size_t dim=128,
	  const double sigma=1.0e-1,	  
	  const std::size_t neg_size=1,
	  const double eta0 = 5e-3 // learning rate	  
	  )
      : num_entries_(), dim_(), neg_size_(), eta0_()
      , vs_(), sqgs_()
    {
      std::vector<std::shared_ptr<DiscreteDistribution> > probs;
      for(const auto&& beg_and_end : begs_and_ends){
	InputIterator beg=beg_and_end.first, end=beg_and_end.second;
	auto prob = std::make_shared<DiscreteDistribution>(beg, end);
	probs.push_back(prob);
      }

      this->initialize(probs, rd, dim, sigma, neg_size, eta0);
    }

    Covec(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
	  std::random_device& rd,	  
	  const std::size_t dim=128,
	  const double sigma=1.0e-1,	  
	  const std::size_t neg_size=1,
	  const double eta0 = 5e-3 // learning rate	  
	  )
      : num_entries_(), dim_(), neg_size_(), eta0_()
      , vs_(), sqgs_()
    { this->initialize(probs, rd, dim, sigma, neg_size, eta0); }
    
  public:

    template <class InputIterator>
    void update_batch(InputIterator beg, InputIterator end, std::random_device& rd);

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

  private:

    void initialize(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
		    std::random_device& rd,		    
		    const std::size_t dim=128,
		    const double sigma=1.0e-1,
		    const std::size_t neg_size=1,
		    const double eta0=5e-3
		    );
    
  private:
    std::vector<std::size_t> num_entries_; // order -> # of entries
    std::size_t dim_;
    std::size_t neg_size_;
    double eta0_;
    std::vector<std::shared_ptr<DiscreteDistribution> > probs_;
    std::vector<std::vector<std::vector<double> > > vs_; // order -> entry -> dim -> value
    std::vector<std::vector<std::vector<double> > > sqgs_; // order -> entry -> dim -> squared gradient
  }; // end of Covectorizer

  // -------------------------------------------------------------------------------

  void Covec::initialize(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
			 std::random_device& rd,		    
			 const std::size_t dim,
			 const double sigma,
			 const std::size_t neg_size,
			 const double eta0
			 )
  {
    probs_ = probs;
    dim_ = dim;
    neg_size_ = neg_size;
    eta0_ = eta0;

    num_entries_.clear();
    vs_.clear();
    sqgs_.clear();

    std::normal_distribution<double> normal(0.0, sigma);
    for(const auto& prob : probs_){
      num_entries_.push_back(prob->probabilities().size());

      std::vector<std::vector<double> > vs, gs;
	
      for(std::size_t i=0; i<prob->probabilities().size(); ++i){
	std::vector<double> v;
	for(std::size_t j=0; j<this->dim_; ++j){
	  v.push_back(normal(rd));
	}
	vs.push_back(v);
	gs.push_back(std::vector<double>(this->dim_, 0.0));
      }

      vs_.push_back(vs);
      sqgs_.push_back(gs);
    }
  }

  template <class InputIterator>
  void Covec::update_batch(InputIterator beg, InputIterator end, std::random_device& rd)
  {
    std::vector<std::unordered_map<std::size_t, std::vector<double> > > grads(this->order());

    // accumulate gradients from positive samples
    std::size_t pos_size = 0;
    for(InputIterator itr = beg; itr != end; ++itr){
      assert(itr->size() == this->order());
      const auto& sample(*itr);

      // compute Hadamard product, inner_product, sigmoid of inner_product
      std::vector<double> Hadamard_product(this->dimension(), 1.0);
      for(std::size_t i=0; i<sample.size(); ++i){
	const auto j = sample[i];
	const auto& v = this->vs_[i][j];
	for(std::size_t k = 0; k < this->dimension(); ++k){
	  Hadamard_product[k] *= v[k];
	}
      }
      double inner_product = std::accumulate(Hadamard_product.begin(), Hadamard_product.end(), 0.0);
      double sigmoid = 1.0 / ( 1 + std::exp(-inner_product) );

      // compute gradients
      for(std::size_t i=0; i<sample.size(); ++i){
	const auto& j = sample[i];
	const auto& v = this->vs_[i][j];
	if(grads[i].find(j) == grads[i].end()){
	  grads[i].insert(std::make_pair(j, std::vector<double>(this->dimension(), 0.0)));
	}
	std::vector<double>& g = grads[i][j];
	for(std::size_t k=0; k < this->dimension(); ++k){
	  if( Hadamard_product[k] == 0 ){ continue; }
	  g[k] += (1-sigmoid) * Hadamard_product[k] / v[k];
	}
      }

    } // end of process for positive sampling


    // accumulate gradients from negative samples
    std::size_t neg_size = this->neg_size() * pos_size;
    for(std::size_t neg_count=0; neg_count < neg_size; ++neg_count){

      // negative sampling
      std::vector<std::size_t> sample(this->order());
      for(std::size_t i=0; i<this->order(); ++i){
	const auto& prob(*this->probs_[i]);
	sample[i] = prob(rd);
      }

      // compute Hadamard product, inner_product, sigmoid of inner_product
      std::vector<double> Hadamard_product(this->dimension(), 1.0);
      for(std::size_t i=0; i<sample.size(); ++i){
	const auto j = sample[i];
	const auto& v = this->vs_[i][j];
	for(std::size_t k = 0; k < this->dimension(); ++k){
	  Hadamard_product[k] *= v[k];
	}
      }
      double inner_product = std::accumulate(Hadamard_product.begin(), Hadamard_product.end(), 0.0);
      double sigmoid = 1.0 / ( 1 + std::exp(-inner_product) );

      // compute gradients
      for(std::size_t i=0; i<sample.size(); ++i){
	const auto& j = sample[i];
	const auto& v = this->vs_[i][j];
	if(grads[i].find(j) == grads[i].end()){
	  grads[i].insert(std::make_pair(j, std::vector<double>(this->dimension(), 0.0)));
	}
	std::vector<double>& g = grads[i][j];
	for(std::size_t k=0; k < this->dimension(); ++k){
	  if( Hadamard_product[k] == 0 ){ continue; }
	  g[k] -= sigmoid * Hadamard_product[k] / v[k];
	}
      }

    } // end of process for negative sampling


    // update
    for(std::size_t i=0; i<this->order(); ++i){

      // update sqgs, vs
      for(const auto& elem : grads[i]){
	const auto j = elem.first;
	const auto& g = elem.second;
	for(std::size_t k=0; k<this->dimension(); ++k){
	  if(g[k] != 0){
	    this->sqgs_[i][j][k] += g[k] * g[k];
	    this->vs_[i][j][k] += this->eta0_ * g[k] / std::sqrt(this->sqgs_[i][j][k]);
	  }
	}
      }

    }
    
    
  } // end of update_batch
  

} // end of namespace covec















