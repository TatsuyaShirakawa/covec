#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <chrono>

#include "covec/covec.hpp"

using namespace covec;
typedef float Real;

inline bool match(const std::string& s, const std::string& longarg, const std::string& shortarg)
{ return std::string(s) == longarg || std::string(s) == shortarg; }

inline bool match(const std::string& s, const std::string& arg)
{ return std::string(s) == arg; }


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

    inline const std::size_t count_of(const std::size_t c) const
    { return this->counts_[c]; }

    inline const std::size_t encode(const std::string& x) const
    { return this->entry2codes_.at(x); }

    inline const std::string decode(const std::size_t c) const
    { return this->code2entries_[c]; }

    // reconstructin encodings by descending order of counts
    void reindex()
    {
      std::vector<std::pair<std::size_t, std::size_t> > ind_and_counts(this->counts_.size());
      std::vector<std::size_t> new_counts(this->counts_.size());
      std::vector<std::string> new_code2entries(this->counts_.size());
      std::unordered_map<std::string, std::size_t> new_entry2codes;

      for(std::size_t i = 0, I = this->counts_.size(); i < I; ++i){
	ind_and_counts[i] = std::make_pair(this->counts_[i], i );
      }

      std::sort(ind_and_counts.begin(), ind_and_counts.end()
		, std::greater<std::pair<std::size_t, std::size_t> >());
      for(std::size_t i = 0, I = ind_and_counts.size(); i < I; ++i){
	std::size_t count = ind_and_counts[i].first,
	  ind = ind_and_counts[i].second;
	const std::string& entry = this->code2entries_[ind];
	new_counts[i] = count;
	new_code2entries[i] = entry;
	new_entry2codes.insert(std::make_pair(entry, i));
      }
      this->counts_ = new_counts;
      this->code2entries_ = new_code2entries;
      this->entry2codes_ = new_entry2codes;
    }

  private:
    std::vector<std::size_t> counts_;
    std::vector<std::string> code2entries_;
    std::unordered_map<std::string, std::size_t> entry2codes_;
  }; // end of CodeBook


  bool load(std::vector<CodeBook>& codebooks,
	    std::vector<std::vector<std::size_t> >& data,
	    const std::string& input_file,
	    const std::size_t order,
	    const char sep,
	    bool sort_enabled
	    )
  {
    codebooks.clear();
    codebooks.resize(2);

    std::ifstream fin(input_file.c_str());
    if(!fin || !fin.good()){
      std::cerr << "cannot open file: " << input_file << std::endl;
      return false;
    }

    std::string line;
    std::size_t n_data = 0;
    while(std::getline(fin, line)){
      ++n_data;
      std::size_t pos_from = 0, pos_to = 0;
      std::vector<std::size_t> instance(order);
      std::size_t i=0;
      do{
	pos_to = line.find_first_of(sep, pos_from);
	std::size_t code = codebooks[i].entry(line.substr(pos_from, pos_to-pos_from));
	instance[i] = code;
	if(pos_to == std::string::npos){
	  pos_from = std::string::npos;
	}else{
	  pos_from = pos_to + 1;
	}
	++i;
	if(i > order){
	  std::cerr << "too many entries in a line: " << line << std::endl;
	  exit(1);
	}
      }while(pos_from != std::string::npos);
      if(i < order){
	if(i > order){
	  std::cerr << "too few entries in a line: " << line << std::endl;
	  exit(1);
	}
      }
      if(!sort_enabled){ data.push_back(instance); }
    }

    if(sort_enabled){
      // renew encodings so that they are sorted by descending order of frequency
      for(std::size_t i = 0; i < order; ++i){
	codebooks[i].reindex();
      }

      // create data
      std::size_t data_idx = data.size();
      data.resize(data.size() + n_data);
      fin.clear(); fin.seekg(0, std::ios_base::beg);
      while(std::getline(fin, line)){
	std::size_t pos_from = 0, pos_to = 0;
	std::vector<std::size_t> instance(order);
	std::size_t i=0;
	do{
	  pos_to = line.find_first_of(sep, pos_from);
	  std::size_t code = codebooks[i].encode(line.substr(pos_from, pos_to-pos_from));
	  instance[i] = code;
	  if(pos_to == std::string::npos){
	    pos_from = std::string::npos;
	  }else{
	    pos_from = pos_to + 1;
	  }
	  ++i;
	}while(pos_from != std::string::npos);
	data[data_idx] = instance;
	++data_idx;
      }
    }      

    
    return true;
  }

  struct Config
  {
    Config()
      : order(2), dim(128), batch_size(512), num_epochs(1)
      , neg_size(1), num_threads(8)
      , sigma(1.0e-1), eta0(5e-3), eta1(1e-5)
      , input_file(), output_prefix("result"), sep('\t')
      , shuffle_enabled(false), sort_enabled(false)
    {}

    std::size_t order;
    std::size_t dim;
    std::size_t batch_size;
    std::size_t num_epochs;
    std::size_t neg_size;
    std::size_t num_threads;
    double sigma;
    double eta0;
    double eta1;
    std::string input_file;
    std::string output_prefix;
    char sep;
    bool shuffle_enabled;
    bool sort_enabled;
  }; // end of Config

  Config parse_args(int narg, const char** argv)
  {
    Config result;
    std::string program_name = argv[0];
    const std::string help_message =
      "usage: " + program_name + " -i input_file [ options ]\n"
      + "Options and arguments:\n"
      "--dim, -d DIM=128                       : the dimension of vectors\n"
      "--batch_size, -b BATCH_SIZE=512         : the (mini)batch size\n"
      "--num_epochs, -n NUM_EPOCHS=1           : the number of epochs\n"
      "--neg_size, -N NEGSIZE=1                : the size of negative sampling\n"
      "--num_threads, -T NUM_THREADS=8         : the number of threads\n"
      "--sigma, -s SIGMA=0.1                   : initialize each element of vector with Normal(0, SIGMA)\n"
      "--eta0, -e ETA0=0.005                   : initial learning rate for SGD\n"
      "--eta1, -E ETA1=0.005                   : final learning rate for SGD\n"
      "--input_file, -i INPUT_FILE             : input file. supposed that each line is separated by SEP\n"
      "--output_prefix, -o OUTPUT_PREFIX=\"vec\" : output file prefix\n"
      "--sep, -S SEP='" "\t" "'                       : separator of each line in INPUT_FILE\n"
      "--help, -h                              : show this help message\n"
      "--shuffle                               : enuable the switch to shuffle data before every epoch\n"
      "--sort                                  : sort entries by descending order of frequency"
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
      }else if( match(argv[i], "--num_threads", "-T") ){
	int x = std::stoi(argv[++i]);
	REQUIRED_POSITIVE(x, "num_threads");
	result.num_threads = static_cast<std::size_t>(x);
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
	if(x == "\\t"){
	  result.sep = '\t';
	}else if(x == "\\s"){
	  result.sep = ' ';
	}else{
	  if( x.length() != 1 ){
	    std::cerr << "sep must be a character but given : " << x << std::endl;
	    exit(1);
	  }
	  result.sep = x[0];
	}
      }else if( match(argv[i], "--shuffle") ){
	result.shuffle_enabled = true;
      }else if( match(argv[i], "--sort") ){
	result.sort_enabled = true;
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
      std::cout << help_message << std::endl;
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


  void save(const std::string& output_prefix, const Covec<Real>& cv, const std::vector<CodeBook>& codebooks)
  {
    // vectors
    const auto& vs = cv.vectors();
    for(std::size_t i = 0, I = cv.order(); i < I; ++i){
      const std::string output_file = output_prefix + "." + std::to_string(i) + ".tsv";
      std::ofstream fout(output_file);
      if(!fout || !fout.good()){
	std::cerr << "cannot open output_vector_file: " << output_file << std::endl;
	exit(1);
      }

      const auto& vs_i = vs[i];
      const auto& codebooks_i = codebooks[i];
      for(std::size_t j = 0, J = vs_i.size(); j < J; ++j){
	const auto& vs_ij = vs_i[j];
	fout << codebooks_i.decode(j);
	for(std::size_t k = 0, K = vs_ij.size(); k < K; ++k){
	  fout << "\t" << vs_ij[k];
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
  const std::size_t num_threads = config.num_threads;
  const double sigma = config.sigma;
  const double eta0 = config.eta0;
  const double eta1 = config.eta1;
  const std::string input_file = config.input_file;
  const std::string output_prefix = config.output_prefix;
  const char sep = config.sep;
  const std::size_t order = detect_order(input_file, sep);
  const bool shuffle_enabled = config.shuffle_enabled;
  const bool sort_enabled = config.sort_enabled;

  std::cout << "config:" << std::endl;
  std::cout << "  " << "dim          : " << dim << std::endl;
  std::cout << "  " <<  "batch_size   : " << batch_size << std::endl;
  std::cout << "  " <<  "num_epochs   : " << num_epochs << std::endl;
  std::cout << "  " <<  "neg_size     : " << neg_size << std::endl;
  std::cout << "  " <<  "num_threads  : " << num_threads << std::endl;  
  std::cout << "  " <<  "sigma        : " << sigma << std::endl;
  std::cout << "  " <<  "eta0         : " << eta0 << std::endl;
  std::cout << "  " <<  "eta1         : " << eta1 << std::endl;
  std::cout << "  " <<  "input_file   : " << input_file << std::endl;
  std::cout << "  " <<  "output_vector_file  : " << output_prefix + ".<#>.tsv" << std::endl;
  std::cout << "  " <<  "sep          : " << "\"" << sep << "\"" << std::endl;
  std::cout << "  " <<  "order        : " << order << std::endl;

  std::vector<CodeBook> codebooks;
  std::vector<std::vector<std::size_t> > data;
  std::mt19937 gen(0);
  std::cout << "loading " << input_file << "..." << std::endl;;
  load(codebooks, data, input_file, order, sep, sort_enabled);

  std::cout << "data size: " << data.size() << std::endl;
  std::cout << "codebook sizes:" << std::endl;
  for(std::size_t i=0; i<order; ++i){
    std::cout << "  " << i << ": " << codebooks[i].size() << std::endl;
  }

  std::cout << "creating distributions..." << std::endl;
  std::vector<std::shared_ptr<std::discrete_distribution<double> > > probs;
  for(std::size_t i=0; i < codebooks.size(); ++i){
    probs.push_back( std::make_shared<std::discrete_distribution<double> >
		     (codebooks[i].counts().begin(), codebooks[i].counts().end())
		     );
  }
  std::cout << "creating covec..." << std::endl;
  Covec<Real> cv(probs, gen, dim, sigma, neg_size, eta0, eta1);
  std::size_t count = 0, cum_count = 0, every_count = 300000;
  auto tick = std::chrono::system_clock::now();
  for(std::size_t epoch=0; epoch<num_epochs; ++epoch){
    std::srand(gen());
    if(shuffle_enabled){ std::random_shuffle(data.begin(), data.end()); }
    for(std::size_t m=0; m < data.size();){

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
      const std::size_t M = std::min(m + batch_size * num_threads, data.size());
      cv.update_batch(data.begin() + m, data.begin() + M, gen, num_threads);
      count += M-m;
      cum_count += M-m;
      m += batch_size * num_threads;
    }
  }
  std::cout << std::endl;
  std::cout << "saving..." << std::endl;
  save(output_prefix, cv, codebooks);

  return 0;
}


