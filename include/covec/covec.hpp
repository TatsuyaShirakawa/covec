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
	  const std::size_t dim=128,
	  const double sigma=1.0e-1,
	  const std::size_t neg_size=1,
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
	  const std::size_t dim=128,
	  const double sigma=1.0e-1,
	  const std::size_t neg_size=1,
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
	  const std::size_t dim=128,
	  const double sigma=1.0e-1,
	  const std::size_t neg_size=1,
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

  private:

    //    constexpr static std::size_t SCALE = (1 << 16);
    constexpr static int SCALE = 1024;
    constexpr static double MAX_VALUE = 20.0;

    template <class RandomGenerator>
    void initialize(const std::vector<std::shared_ptr<DiscreteDistribution> >& probs,
		    RandomGenerator& gen,
		    const std::size_t dim=128,
		    const double sigma=1.0e-1,
		    const std::size_t neg_size=1,
		    const double eta0=5e-3,
		    const double eta1=1e-5
		    );


    template <class Grad, class InputIterator>
    void compute_positive_grad(Grad& grads, InputIterator beg, InputIterator end, std::size_t& pos_size);

    template <class Grad, class RandomGenerator>
    void compute_negative_grad(Grad& grads, RandomGenerator& gen, std::size_t neg_size);


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
    for(std::size_t i=0; i<order; ++i){
      const auto& prob = probs_[i];
      num_entries_.push_back(prob->probabilities().size());

      std::size_t num_entries = prob->probabilities().size();
      std::vector<std::vector<double> > vs(num_entries);

      for(std::size_t j=0; j < num_entries; ++j){
	std::vector<double> v(this->dim_);
	for(std::size_t k=0; k<this->dim_; ++k){
	  v[k] = normal(gen);
	}
	vs[j] = v;
      }

      this->vs_[i] = vs;
      this->cs_[i] = std::vector<std::size_t>(num_entries, 0);
    }
  }

  template <class Grad, class InputIterator>
  void Covec::compute_positive_grad(Grad& grads, InputIterator beg, InputIterator end, std::size_t& pos_size)
  {
    pos_size = 0;
    for(InputIterator itr = beg; itr != end; ++itr){

      ++pos_size;

      assert(itr->size() == this->order());
      const auto& sample(*itr);
      assert( sample.size() == this->order() );

      // count the occurences and 
      // compute Hadamard product, inner_product, sigmoid of inner_product
      std::vector<double> Hadamard_product(this->dimension(), 1.0);
      for(std::size_t i=0; i<this->order(); ++i){
	const auto j = sample[i];
	++this->cs_[i][j];
	const auto& v = this->vs_[i][j];
	assert( v.size() == this->dimension() );
	for(std::size_t k = 0; k < this->dimension(); ++k){
	  Hadamard_product[k] *= v[k];
	}
      }
      double inner_product = std::accumulate(Hadamard_product.begin(), Hadamard_product.end(), 0.0);
      double sigmoid = 1.0 / ( 1 + std::exp(-inner_product) );

      // compute gradients
      for(std::size_t i=0; i<this->order(); ++i){
	const auto& j = sample[i];
	const auto& v = this->vs_[i][j];
	auto itr = grads[i].find(j);
	if(itr == grads[i].end()){
	  auto ret = grads[i].insert(std::make_pair(j, std::vector<double>(this->dimension(), 0.0)));
	  itr = ret.first;
	}
	//	std::vector<double>& g = grads[i][j];
	std::vector<double>& g = itr->second;
	for(std::size_t k=0; k < this->dimension(); ++k){
	  if( Hadamard_product[k] == 0 ){ continue; }
	  g[k] += (1-sigmoid) * Hadamard_product[k] / v[k];
	}
      }

    } // end of process for positive sampling
  }

  template <class Grad, class RandomGenerator>
  void Covec::compute_negative_grad(Grad& grads, RandomGenerator& gen, std::size_t neg_size)
  {
    for(std::size_t neg_count=0; neg_count < neg_size; ++neg_count){

      // negative sampling and
      // count the occurences
      std::vector<std::size_t> sample(this->order());
      for(std::size_t i=0; i<this->order(); ++i){
	const auto& prob(*this->probs_[i]);
	sample[i] = prob(gen);
	++this->cs_[i][sample[i]];
      }
      assert( sample.size() == this->order() );

      // compute Hadamard product, inner_product, sigmoid of inner_product
      std::vector<double> Hadamard_product(this->dimension(), 1.0);
      for(std::size_t i=0; i<this->order(); ++i){
	const auto j = sample[i];
	const auto& v = this->vs_[i][j];
	for(std::size_t k = 0; k < this->dimension(); ++k){
	  Hadamard_product[k] *= v[k];
	}
      }
      double inner_product = std::accumulate(Hadamard_product.begin(), Hadamard_product.end(), 0.0);
      double sigmoid = 1.0 / ( 1 + std::exp(-inner_product) );

      // compute gradients
      for(std::size_t i=0; i<this->order(); ++i){
	const auto& j = sample[i];
	const auto& v = this->vs_[i][j];
	auto itr = grads[i].find(j);
	if(itr == grads[i].end()){
	  auto ret = grads[i].insert(std::make_pair(j, std::vector<double>(this->dimension(), 0.0)));
	  itr = ret.first;
	}
	std::vector<double>& g = itr->second;
	for(std::size_t k=0; k < this->dimension(); ++k){
	  if( Hadamard_product[k] == 0 ){ continue; }
	  g[k] -= sigmoid * Hadamard_product[k] / v[k];
	}
      }

    } // end of process for negative sampling
  }


  template <class InputIterator, class RandomGenerator>
  void Covec::update_batch(InputIterator beg, InputIterator end, RandomGenerator& gen)
  {
    std::vector<std::unordered_map<std::size_t, std::vector<double> > > grads(this->order());

    // accumulate gradients from positive samples
    std::size_t pos_size = 0;
    compute_positive_grad(grads, beg, end, pos_size);

    // accumulate gradients from negative samples
    std::size_t neg_size = this->neg_size() * pos_size;
    compute_negative_grad(grads, gen, neg_size);

    // update
    for(std::size_t i=0; i<this->order(); ++i){
      auto& vs_i = this->vs_[i];
      const auto& cs_i = this->cs_[i];
      // update sqgs, vs
      for(const auto& elem : grads[i]){
	const auto j = elem.first;
	const auto& g = elem.second;
	auto& vs_ij = vs_i[j];
	for(std::size_t k=0; k<this->dimension(); ++k){
	  if(g[k] != 0){
	    double eta = std::max(this->eta0_ / cs_i[j], this->eta1_);
	    vs_ij[k] += eta * g[k];
	  }
	}
      }

    }


  } // end of update_batch

} // end of namespace covec

