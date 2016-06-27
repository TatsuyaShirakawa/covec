#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>

#include "covec/covec.hpp"

using namespace covec;

namespace{

  struct CodeBook
{
  CodeBook(): counts_(), code2entries_(), entry2codes_() {}

  const std::size_t entry(const std::string& x)
  {
    auto itr = this->entry2codes_.find(x);
    if(itr == this->entry2codes_.end()){
      auto entried = this->entry2codes_.insert(std::make_pair(x, code2entries_.size()));
      itr = entried.first;
      this->code2entries_.push_back(x);
      this->counts_.push_back(0);
    }
    ++this->counts_[itr->second];
    return itr->second;
  }

  inline const std::vector<std::size_t>& counts() const
  { return this->counts_; }

  inline const std::size_t encode(const std::string& x) const
  { return this->entry2codes_.at(x); }

  inline const std::string decode(const std::size_t c) const
  { return this->code2entries_[c]; }
  
private:
  std::vector<std::size_t> counts_;
  std::vector<std::string> code2entries_;
  std::unordered_map<std::string, std::size_t> entry2codes_;
}; // end of CodeBook


  bool load(std::vector<CodeBook>& codebooks,
	    std::vector<std::vector<std::size_t> >& data,
	    const std::string& input_file,
	    const char sep='\n',
	    const std::size_t order=2)
  {
    codebooks.clear();
    codebooks.resize(2);
    
    std::ifstream fin(input_file.c_str());
    if(!fin || !fin.good()){
      std::cerr << "cannot open file: " << input_file << std::endl;
      return false;
    }

    std::string line;
    while(std::getline(fin, line, sep)){
      std::stringstream sin;
      sin << line;
      std::vector<std::size_t> instance(order);
      for(std::size_t i=0; i<order; ++i){
	std::string x;
	sin >> x;
	std::size_t code = codebooks[i].entry(x);
	instance[i] = code;
      }
      data.push_back(instance);
    }
    
    return true;
  }

}


int main()
{
  std::size_t batch_size = 8;
  std::size_t num_epochs = 100;
  
  std::vector<CodeBook> codebooks;
  std::vector<std::vector<std::size_t> > data;
  std::random_device rd;
  load(codebooks, data, "../data/coffee.tsv");

  std::vector<std::shared_ptr<DiscreteDistribution> > probs;
  for(std::size_t i=0; i<codebooks.size(); ++i){
    probs.push_back( std::make_shared<DiscreteDistribution>
		     (codebooks[i].counts().begin(), codebooks[i].counts().end())
		     );
  }
  Covec cv(probs, rd, 10);

  for(std::size_t epoch=0; epoch<num_epochs; ++epoch){
    for(std::size_t m=0; m < data.size(); m += batch_size){
      const std::size_t M = std::min(m + batch_size, data.size());
      cv.update_batch(data.begin() + m, data.begin() + M, rd);
    }
  }


  for(std::size_t i=0; i<cv.order(); ++i){
    std::cout << "order " << i << std::endl;
    const auto& vi = cv.vectors()[i];
    for(std::size_t j=0; j<vi.size(); ++j){
      std::cout << "\t" << codebooks[i].decode(j) << " :";
      const auto& vij = vi[j];
      for(std::size_t k=0; k<vij.size(); ++k){
	std::cout << " " << vij[k];
      }
      std::cout << std::endl;
    }
  }
  
    
  return 0;
}


