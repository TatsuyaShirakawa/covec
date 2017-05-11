// Copyright 2016 Tatsuya Shirakawa. All Rights Reserved.
#pragma once

#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>
#include <memory>
#include <random>
#include <unordered_map>
#include <thread>

namespace covec{

  template <class Real=double>
  struct Covec
  {
  public:

    template <class RandomGenerator>
    Covec(const std::vector<std::vector<std::size_t> >& each_counts,
          RandomGenerator& gen,
          const std::size_t dim = 128,
          const Real sigma = 1.0e-1,
          const std::size_t neg_size = 1,
          const Real eta0 = 5e-3, // initial learning rate
          const Real eta1 = 1e-5,  // final learning rate
          const bool shared = false // if true vectors of each order are shared
          )
      : num_entries_(), dim_(), neg_size_(), eta0_(), eta1_()
      , vs_(), cs_(), shared_()
    {
      std::vector<std::shared_ptr<std::discrete_distribution<int> > > probs;
      for(const auto& counts : each_counts){
        auto prob = std::make_shared<std::discrete_distribution<int> >(counts.begin(), counts.end());
        probs.push_back(prob);
      }
      this->initialize(probs, gen, dim, sigma, neg_size, eta0, eta1, shared);
    }

    template <class InputIterator, class RandomGenerator>
    Covec(const std::vector<std::pair<InputIterator, InputIterator> >& begs_and_ends,
          RandomGenerator& gen,
          const std::size_t dim = 128,
          const Real sigma = 1.0e-1,
          const std::size_t neg_size = 1,
          const Real eta0 = 5e-3, // learning rate
          const Real eta1 = 1e-5, // final learning rate
          bool shared = false // if true vectors of each order are shared	  
          )
      : num_entries_(), dim_(), neg_size_(), eta0_(), eta1_()
      , vs_(), cs_(), shared_()
    {
      std::vector<std::shared_ptr<std::discrete_distribution<int> > > probs;
      for(const auto&& beg_and_end : begs_and_ends){
        InputIterator beg=beg_and_end.first, end=beg_and_end.second;
        auto prob = std::make_shared<std::discrete_distribution<int> >(beg, end);
        probs.push_back(prob);
      }
      this->initialize(probs, gen, dim, sigma, neg_size, eta0, eta1, shared);
    }

    template <class RandomGenerator>
    Covec(const std::vector<std::shared_ptr<std::discrete_distribution<int> > >& probs,
          RandomGenerator& gen,
          const std::size_t dim = 128,
          const Real sigma = 1.0e-1,
          const std::size_t neg_size = 1,
          const Real eta0 = 5e-3, // learning rate
          const Real eta1 = 1e-5, // final learning rate
          bool shared = false // if true vectors of each order are shared	  
          )
      : num_entries_(), dim_(), neg_size_(), eta0_()
      , vs_(), cs_(), shared_()
    { this->initialize(probs, gen, dim, sigma, neg_size, eta0, eta1, shared); }

  public:

    template <class InputIterator, class RandomGenerator>
    void update_batch(InputIterator beg, InputIterator end, RandomGenerator& gen);
		      

    template <class InputIterator, class GradIterator, class RandomGenerator>
    void update_batch_thread(InputIterator beg, InputIterator end, GradIterator gbeg, RandomGenerator& gen);    

  public:

    inline const std::size_t order() const
    { return this->num_entries_.size(); }

    inline const std::vector<std::shared_ptr<std::size_t> >& num_entries() const
    { return this->num_entries_; }

    inline const std::vector<std::shared_ptr<std::vector<std::vector<Real> > > >& vectors() const
    { return this->vs_; }

    inline const std::size_t dimension() const
    { return this->dim_; }

    inline const std::size_t neg_size() const
    { return this->neg_size_; }

    inline const Real eta0() const
    { return this->eta0_; }

    inline const Real eta1() const
    { return this->eta1_; }

    inline bool shared() const
    { return this->shared_; }

  private:

    template <class RandomGenerator>
    void initialize(const std::vector<std::shared_ptr<std::discrete_distribution<int> > >& probs,
                    RandomGenerator& gen,
                    const std::size_t dim = 128,
                    const Real sigma = 1.0e-1,
                    const std::size_t neg_size = 1,
                    const Real eta0 = 5e-3,
                    const Real eta1 = 1e-5,
                    bool shared = false // if true vectors of each order are shared		    
                    );

    typedef enum { POSITIVE, NEGATIVE } POS_NEG;

    template <class Grad>
    void accumulate_grad(Grad& grad
                         , const std::vector<std::size_t>& sample
                         , const POS_NEG pos_neg);

    // in case order == 2
    template <class Grad>
    void accumulate_grad_2(Grad& grad
                           , const std::vector<std::size_t>& sample
                           , const POS_NEG pos_neg);

    inline Real next_eta(const std::size_t count) const
    {
      constexpr std::size_t C = 100000;
      if(count < C){
        auto delta = this->eta0_ - this->eta1_;
        auto eta = this->eta1_ + (delta * (C - count)) / C;
        return eta;
      }else{
        return this->eta1_;
      }
    }

  private:
    std::vector<std::shared_ptr<std::size_t> > num_entries_; // order -> # of entries
    std::size_t dim_;
    std::size_t neg_size_;
    Real eta0_;
    Real eta1_;
    std::vector<std::shared_ptr<std::discrete_distribution<int> > > probs_;
    //    std::vector<std::vector<std::vector<Real> > > vs_; // order -> entry -> dim -> value
    std::vector<std::shared_ptr<std::vector<std::vector<Real> > > > vs_; // order -> entry -> dim -> value    
    std::vector<std::shared_ptr<std::vector<std::size_t> > > cs_; // counts of occurrencees
    bool shared_;
  }; // end of Covectorizer

  // -------------------------------------------------------------------------------

  template <class Real>
  template <class RandomGenerator>
  void Covec<Real>::initialize(const std::vector<std::shared_ptr<std::discrete_distribution<int> > >& probs,
                               RandomGenerator& gen,
                               const std::size_t dim,
                               const Real sigma,
                               const std::size_t neg_size,
                               const Real eta0,
                               const Real eta1,
                               const bool shared
                               )
  {
    this->probs_ = probs;
    this->dim_ = dim;
    this->neg_size_ = neg_size;
    this->eta0_ = eta0;
    this->eta1_ = eta1;
    this->shared_ = shared;
    
    this->num_entries_.clear();
    this->vs_.clear();
    this->cs_.clear();

    std::size_t order = this->probs_.size();
    this->vs_.resize(order);
    this->cs_.resize(order);
    this->num_entries_.resize(order);
    std::normal_distribution<Real> normal(0.0, sigma);
    for(std::size_t i = 0; i < order; ++i){
      if(this->shared() && i > 0){
        assert(this->probs_[i] == this->probs_[0]);
        this->vs_[i] = this->vs_[0];
        this->num_entries_[i] = this->num_entries_[0];
        this->cs_[i] = this->cs_[0];
      }else{
        const auto& prob = this->probs_[i];

        // allocate
        this->num_entries_[i] =
          std::make_shared<std::size_t>(prob->probabilities().size());
        std::size_t num_entries = prob->probabilities().size();
        this->vs_[i] =
          std::make_shared<std::vector<std::vector<Real> > >
          (num_entries, std::vector<Real>(this->dimension()));
        this->cs_[i] =
          std::make_shared<std::vector<std::size_t> >(num_entries);
	   
        auto& vs = *this->vs_[i];
        vs.resize(num_entries);
        for(std::size_t j = 0; j < num_entries; ++j){
          std::vector<Real> v(this->dim_);
          for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
            v[k] = normal(gen);
          }
          vs[j] = v;
        }
        this->cs_[i] = std::make_shared<std::vector<std::size_t> >(num_entries, 0);
      }
    }
  }

  template <class Real>
  template <class Grad>
  void Covec<Real>::accumulate_grad(Grad& grad
                                    , const std::vector<std::size_t>& sample
                                    , const typename Covec::POS_NEG pos_neg)
  {
    assert( sample.size() == this->order() );

    if( this->order() == 2 ){
      return accumulate_grad_2(grad, sample, pos_neg);
    }

    // count the occurences and
    // compute Hadamard product, inner_product, sigmoid of inner_product
    std::vector<Real> Hadamard_product(this->dimension(), 1.0);
    for(std::size_t i = 0, I = this->order(); i < I; ++i){
      const auto j = sample[i];
      ++(*this->cs_[i])[j];
      const auto& v = (*this->vs_[i])[j];
      assert( v.size() == this->dimension() );
      for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
        Hadamard_product[k] *= v[k];
      }
    }
    Real inner_product =
      static_cast<Real>(std::accumulate(Hadamard_product.begin(), Hadamard_product.end(), 0.0));
    Real sigmoid = static_cast<Real>( 1.0 / ( 1 + std::exp(-inner_product) ) );
    Real coeff = (pos_neg == POSITIVE ? 1 - sigmoid : -sigmoid);

    // compute gradients
    for(std::size_t i = 0, I = this->order(); i < I; ++i){
      const auto& j = sample[i];
      auto& cs_ij = (*this->cs_[i])[j];
      ++cs_ij;
      const auto& v = (*this->vs_[i])[j];
      auto& grad_i = grad[i];
      grad_i.first = j;
      auto& g = grad_i.second;
      auto eta = this->next_eta(cs_ij);
      const auto C = coeff * eta;
      for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
        if( Hadamard_product[k] == 0 ){ continue; }
        g[k] = C * Hadamard_product[k] / v[k];
      }
    }

  }

  /** optimized ver. in case order == 2 */
  template <class Real>
  template <class Grad>
  void Covec<Real>::accumulate_grad_2(Grad& grad
                                      , const std::vector<std::size_t>& sample
                                      , const typename Covec::POS_NEG pos_neg)
  {
    assert( sample.size() == 2 && this->order() == 2 );

    // count the occurences and
    // compute Hadamard product, inner_product, sigmoid of inner_product
    const auto& v0 = (*this->vs_[0])[sample[0]];
    const auto& v1 = (*this->vs_[1])[sample[1]];
    const Real inner_product =
      static_cast<Real>(std::inner_product(v0.begin(), v0.end(), v1.begin(), 0.0));
    Real sigmoid = static_cast<Real>( 1.0 / ( 1 + std::exp(-inner_product) ) );
    Real coeff = (pos_neg == POSITIVE ? 1 - sigmoid : -sigmoid);

    for(std::size_t i = 0; i < 2; ++i){
      const std::size_t j = sample[i];
      auto& cs_ij = (*this->cs_[i])[j];
      ++cs_ij;
      assert( grad.size() == this->order() );
      auto& grad_i = grad[i];
      grad_i.first = j;
      assert( grad_i.second.size() == this->dimension() );      
      auto& g = grad_i.second;
      const auto& v = (i == 0 ? v1 : v0);
      auto eta = this->next_eta(cs_ij);
      const auto C = coeff * eta;
      for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
        g[k] = C * v[k];
      }
    }
  }

  template <class Real>
  template <class InputIterator, class GradIterator, class RandomGenerator>
  void Covec<Real>::update_batch_thread(InputIterator beg, InputIterator end,
                                        GradIterator gbeg,
                                        RandomGenerator& gen)
  {
    // accumulate gradients
    std::vector<std::size_t> negative_sample(this->order());
    
    auto gitr = gbeg;
    for(auto itr = beg; itr != end; ++itr){
      // gradient from positive sample
      assert(gitr->size() == order() );
      assert( itr->size() == order() );
      accumulate_grad(*gitr, *itr, POSITIVE);
      ++gitr;
      // gradient(s) from negative sample      
      for(std::size_t m = 0, M = this->neg_size(); m < M; ++m){
        for(std::size_t i = 0, I = this->order(); i < I; ++i){
          std::size_t j = this->probs_[i]->operator()(gen);
          negative_sample[i] = j;
        }
        assert( negative_sample.size() == order() );
        accumulate_grad(*gitr, negative_sample, NEGATIVE);
        ++gitr;
      }
    }
    auto gend = gitr;      
    
    // update
    for(gitr = gbeg; gitr != gend; ++gitr){
      const auto& grad = *gitr;
      for(std::size_t i = 0, I = this->order(); i < I; ++i){
        auto& vs_i = *this->vs_[i];
        const auto& grad_i = grad[i];
        // update sqgs, vs
        const auto j = grad_i.first;
        const auto& grad_ij = grad_i.second;
        auto& vs_ij = vs_i[j];
        for(std::size_t k = 0, K = this->dimension(); k < K; ++k){
          vs_ij[k] += grad_ij[k];
        }
      }
    }
    
  }

  
  template <class Real>
  template <class InputIterator, class RandomGenerator>
  void Covec<Real>::update_batch(InputIterator beg, InputIterator end,
                                 RandomGenerator& gen)
  {
    typedef std::pair<std::size_t, std::vector<Real> > j_grad;
    //    static std::vector< std::vector< j_grad > > grads; // data_idx -> order -> entry -> dim -> value
    static std::vector< std::vector< j_grad > > grads; // data_idx -> order -> entry -> dim -> value    

    // reserve sizes of grads
    std::size_t pos_data_size = static_cast<std::size_t>(std::distance(beg, end));
    std::size_t data_size = (1 + this->neg_size()) * pos_data_size; 

    if(grads.size() < data_size){
      grads.resize(data_size,
                   std::vector<j_grad>( this->order(), j_grad( 0, std::vector<Real>(this->dimension()) ) )
                   );
    }

    // single thread
    update_batch_thread(beg, end, grads.begin(), gen);

  } // end of update_batch

} // end of namespace covec











