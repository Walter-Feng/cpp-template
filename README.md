# CPP template

This is a minimal implementation of a modern C++ project.


The repository gives examples for the following:

1. CMake
2. Using submodules (Catch2, args, json)
3. Finding library (HDF5 but not required to compile)
4. Enabling OpenMP 
5. Git versioning (i.e. automatic versioning by git hash)
6. Creating a library
7. Creating an executable
8. C++ templates, concepts (C++-20 feature)
9. Creating unit tests
10. Error handling

with my taste on software engineering (e.g. naming convention and file structure).

This can be used as a template for your next C++ project, so that you don't need
to copy the same snippets again. Make sure to change the name to your own project -
you will be replacing `TEMPLATE / Template / template` names except the 
actual `template` keyword for C++ in `src/util/vectorization.h`.

To compile, you will be doing:
1. clone the repository with all the submodules
```
git clone --recursive https://github.com:Walter-Feng/cpp-template.git
```
2. make a build directory
```
cd cpp-template
mkdir build
cd build
```
3. run cmake
```
cmake ..
```
4. make
```
make -j
```
5. run the executable
```
./template
```
6. run the unit test
```
test/test_template
```