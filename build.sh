export BUILD_DIR=build

if [ ! -d $BUILD_DIR ]; then
  mkdir $BUILD_DIR
fi

g++ -Wall -O3 -std=c++11 -lpthread -Iinclude src/covec.cpp -o $BUILD_DIR/covec 
