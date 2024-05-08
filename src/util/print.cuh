#ifndef CPP_TEMPLATE_UTIL_PRINT_CUH
#define CPP_TEMPLATE_UTIL_PRINT_CUH

#include <iostream>
#include <cuda_runtime.h>

namespace cpp_template::util {

template<typename T>
void print_cuda(const T * src, const size_t total_length,
                const std::string header = "") {
  std::vector<T> converted(total_length);
  cudaMemcpy(converted.data(), src, sizeof(T) * total_length,
             cudaMemcpyDeviceToHost);

  std::cout << header;

  for(const auto element : converted) {
    std::cout << element << " ";
  }

  std::cout << std::endl;
}
}

#endif //CPP_TEMPLATE_UTIL_PRINT_CUH
