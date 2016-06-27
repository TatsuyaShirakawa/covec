export BUILD_DIR=build

if [ ! -d $BUILD_DIR ]; then
  mkdir $BUILD_DIR
fi

g++ -Wall -O2 -std=c++14 -Iinclude src/covec.cpp -o $BUILD_DIR/covec
