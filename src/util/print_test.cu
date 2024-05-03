#include <catch2/catch_test_macros.hpp>

#include "print.cuh"

TEST_CASE("Check print cuda vector") {

  const std::vector<double> host_vector{1, 1, 4, 5, 1, 4};
  double * device_vector;
  cudaMalloc(&device_vector, sizeof(double) * host_vector.size());
  cudaMemcpy(device_vector, host_vector.data(),
             sizeof(double) * host_vector.size(), cudaMemcpyHostToDevice);

  cpp_template::util::print_cuda(device_vector, host_vector.size(),
                                 "test vector print: ");
  cudaFree(device_vector);

}