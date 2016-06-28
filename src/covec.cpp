#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <random>
#include <iomanip>
#include <unordered_map>
#include <chrono>

#include "covec/covec.hpp"

using namespace covec;

inline bool match(const std::string& s, const std::string& longarg, const std::string& shortarg)
{ return std::string(s) == longarg || std::string(s) == shortarg; }


#define REQUIRED_POSITIVE(x, name)					\
  if(x <= 0){								\
    if( x <= 0 ){							\
      std::cerr << name " must be > 0 but given: " << x << std::endl;	\
      exit(1);								\
    }									\
  }									\
  

namespace{

  struct CodeBook
  {
    CodeBook(): counts_(), code2entries_(), entry2codes_() {}

    const std::size_t entry(const std::string& x)
    {
      auto itr = this->entry2codes_.find(x);
      if(itr == this->entry2codes_.end()){
	auto entried = this->entry2codes_.insert
	  (std::make_pair(x, code2entries_.size()));
	itr = entried.first;
	this->code2entries_.push_back(x);
	this->counts_.push_back(0);
      }
      ++this->counts_[itr->second];
      return itr->second;
    }

    inline const std::size_t size() const
    { return this->code2entries_.size(); }

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
	    const std::size_t order)
  {
    codebooks.clear();
    codebooks.resize(2);
    
    std::ifstream fin(input_file.c_str());
    if(!fin || !fin.good()){
      std::cerr << "cannot open file: " << input_file << std::endl;
      return false;
    }

    std::string line;
    while(std::getline(fin, line)){
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

  struct Config
  {
    Config()
      : order(2), dim(128), batch_size(32), num_epochs(1)
      , neg_size(1), sigma(1.0e-1), eta0(5e-3)
      , input_file(), output_prefix("covec"), sep('\t')
    {}

    std::size_t order;
    std::size_t dim;
    std::size_t batch_size;
    std::size_t num_epochs;
    std::size_t neg_size;
    double sigma;
    double eta0;
    std::string input_file;
    std::string output_prefix;
    char sep;
  }; // end of Config

  Config parse_args(int narg, const char** argv)
  {
    Config result;
    std::string program_name = argv[0];
    const std::string help_message =
      "usage: " + program_name + " -i input_file [ options ]\n"
      + "Options and arguments:\n"
      "--dim, -d DIM=128                        : the dimension of vectors\n"
      "--batch_size, -b BATCH_SIZE=32           : BATCH_SIZE: the (mini)batch size\n"
      "--num_epochs, -n NUM_EPOCHS=1            : NUM_EPOCHS: the number of epochs\n"
      "--neg_size, -N NEGSIZE=1                 : the size of negative sampling\n"
      "--sigma, -s SIGMA=0.1                    : initialize each element of vector with Normal(0, SIGMA)\n"
      "--eta0, -e ETA0=0.005                    : initial learning rate for AdaGrad\n"
      "--input_file, -i INPUT_FILE              : input file. supposed that each line is separated by SEP\n"
      "--output_prefix, -o OUTPUT_PREFIX=\"covec\": output file prefix\n"
      "--sep, -s SEP='\t'                       : separator of each line in INPUT_FILE\n"
      "--help, -h                               : show this help message\n"
      ;
    bool input_file_found = false;
    for(int i=1; i<narg; ++i){
      if( match(argv[i], "--dim", "-d") ){
	int x = std::stoi(argv[++i]);
	REQUIRED_POSITIVE(x, "dim");
	result.dim = static_cast<std::size_t>(x);
      }else if( match(argv[i], "--batch_size", "-b") ){
	int x = std::stoi(argv[++i]);
	REQUIRED_POSITIVE(x, "batch_size");
	result.batch_size = static_cast<std::size_t>(x);
      }else if( match(argv[i], "--num_epochs", "-n") ){
	int x = std::stoi(argv[++i]);
	REQUIRED_POSITIVE(x, "num_epochs");
	result.num_epochs = static_cast<std::size_t>(x);
      }else if( match(argv[i], "--neg_size", "-N") ){
	int x = std::stoi(argv[++i]);
	REQUIRED_POSITIVE(x, "neg_size");
	result.neg_size = static_cast<std::size_t>(x);
      }else if( match(argv[i], "--sigma", "-s") ){
	double x = std::stod(argv[++i]);
	REQUIRED_POSITIVE(x, "sigma");
	result.sigma = x;
      }else if( match(argv[i], "--eta0", "-e") ){
	double x = std::stod(argv[++i]);
	REQUIRED_POSITIVE(x, "eta0");
	result.eta0 = x;
      }else if( match(argv[i], "--input_file", "-i") ){
	input_file_found = true;
	std::string x = argv[++i];
	result.input_file = x;
      }else if( match(argv[i], "--sep", "-S") ){
	std::string x = argv[++i];
	if( x.length() != 1 ){
	  std::cerr << "sep must be a character but given : " << x << std::endl;
	  exit(1);
	}
	result.sep = x[0];
      }else if( match(argv[i], "--help", "-h") ){
	std::cout << help_message << std::endl;
	exit(0);
      }else{
	std::cerr << "invalid argument: " << argv[i] << std::endl;
	std::cerr << help_message << std::endl;
	exit(1);
      }
    }

    if( !input_file_found ){
      std::cerr << "input_file required" << std::endl;
      exit(1);
    }
    return result;
  } // end of parse_args

  std::size_t detect_order(const std::string& input_file, const char sep=' ')
  {
    std::ifstream fin(input_file);
    if(!fin || !fin.good()){
      std::cerr << "input file cannot open : " << input_file << std::endl;
      exit(1);
    }
    std::string line;
    if(!std::getline(fin, line)){
      std::cerr << "cannot read a line from input file : " << input_file << std::endl;
      exit(1);
    }
    if(line == ""){
      std::cerr << "empty first line in  input file : " << input_file << std::endl;
      exit(1);
    }
    std::size_t result=1;
    std::size_t pos=0;
    while( (pos = line.find(sep, pos)) != std::string::npos ){
      ++result;
      ++pos;
    }
    return result;
  }


  void save(const std::string& output_prefix, const Covec& cv, const std::vector<CodeBook>& codebooks)
  {
    // codebooks
    for(std::size_t i=0; i<cv.order(); ++i){
      const std::string output_file = output_prefix + "." + std::to_string(i) + ".codebook.tsv";
      std::ofstream fout(output_file);
      if(!fout || !fout.good()){
	std::cerr << "cannot open output_codebook_file: " << output_file << std::endl;
	exit(1);
      }
      
      for(std::size_t j=0; j<codebooks[i].size(); ++j){
	fout << j << "\t" << codebooks[i].decode(j) << "\n";
      }
    }

    // vectors
    const auto& vs = cv.vectors();
    for(std::size_t i=0; i<cv.order(); ++i){
      const std::string output_file = output_prefix + "." + std::to_string(i) + ".vector.tsv";
      std::ofstream fout(output_file);
      if(!fout || !fout.good()){
	std::cerr << "cannot open output_vector_file: " << output_file << std::endl;
	exit(1);
      }
      
      for(std::size_t j=0; j<vs[i].size(); ++j){
	fout << j;
	for(std::size_t k=0; k<cv.dimension(); ++k){
	  fout << "\t" << vs[i][j][k];
	}
	fout << "\n";
      }
    }
    
  } // end of save

} // end of anonymous namespace


int main(int narg, const char** argv)
{
  const auto& config = parse_args(narg, argv);

  const std::size_t dim = config.dim;
  const std::size_t batch_size = config.batch_size;
  const std::size_t num_epochs = config.num_epochs;
  const std::size_t neg_size = config.neg_size;
  const double sigma = config.sigma;
  const double eta0 = config.eta0;
  const std::string input_file = config.input_file;
  const std::string output_prefix = "./result";
  const char sep = config.sep;
  const std::size_t order = detect_order(input_file, sep);

  std::cout << "config:" << std::endl;
  std::cout << "  " << "dim          : " << dim << std::endl;
  std::cout << "  " <<  "batch_size   : " << batch_size << std::endl;
  std::cout << "  " <<  "num_epochs   : " << num_epochs << std::endl;
  std::cout << "  " <<  "neg_size     : " << neg_size << std::endl;
  std::cout << "  " <<  "sigma        : " << sigma << std::endl;
  std::cout << "  " <<  "eta0         : " << eta0 << std::endl;
  std::cout << "  " <<  "input_file   : " << input_file << std::endl;
  std::cout << "  " <<  "output_codebook_file: " << output_prefix + ".<#>.codebook.tsv" << std::endl;
  std::cout << "  " <<  "output_vector_file  : " << output_prefix + ".<#>.vector.tsv" << std::endl;  
  std::cout << "  " <<  "sep          : " << "\"" << sep << "\"" << std::endl;
  std::cout << "  " <<  "order        : " << order << std::endl;
  
  std::vector<CodeBook> codebooks;
  std::vector<std::vector<std::size_t> > data;
  std::random_device rd;
  std::cout << "loading " << input_file << "..." << std::endl;;
  load(codebooks, data, input_file, order);

  std::cout << "data size: " << data.size() << std::endl;  
  std::cout << "codebook sizes:" << std::endl;
  for(std::size_t i=0; i<order; ++i){
    std::cout << "  " << i << ": " << codebooks[i].size() << std::endl;
  }

  std::cout << "creating distributions..." << std::endl;
  std::vector<std::shared_ptr<DiscreteDistribution> > probs;
  for(std::size_t i=0; i < codebooks.size(); ++i){
    probs.push_back( std::make_shared<DiscreteDistribution>
		     (codebooks[i].counts().begin(), codebooks[i].counts().end())
		     );
  }
  std::cout << "creating covec..." << std::endl;
  Covec cv(probs, rd, dim, sigma, neg_size, eta0);

  std::size_t count = 0, cum_count = 0, every_count = 10000;
  auto tick = std::chrono::system_clock::now();
  for(std::size_t epoch=0; epoch<num_epochs; ++epoch){
    std::random_shuffle(data.begin(), data.end());
    for(std::size_t m=0; m < data.size(); m += batch_size){
      
      if(count >= every_count){ // reporting
	auto tack = std::chrono::system_clock::now();
	auto millisec = std::chrono::duration_cast<std::chrono::milliseconds>(tack - tick).count();
	double percent = (cum_count * 100.0) / (data.size() * num_epochs);
	std::size_t words_per_sec = (1000*count) / millisec;
	std::cout << "\r"
		  << "epoch " << std::right << std::setw(3) << epoch+1 << "/" << num_epochs
		  << "  " << std::left << std::setw(5) << std::fixed << std::setprecision(2) << percent << " %"
		  << "  " << std::left << std::setw(6) << words_per_sec << " words/sec."
		  << std::flush;

	count = 0;
	tick = std::chrono::system_clock::now();
      }
      
      const std::size_t M = std::min(m + batch_size, data.size());
      cv.update_batch(data.begin() + m, data.begin() + M, rd);
      count += M-m;
      cum_count += M-m;
    }
  }
  std::cout << "saving..." << std::endl;
  save(output_prefix, cv, codebooks);
  
  return 0;
}


