#include <catch2/catch_test_macros.hpp>

#include "util.h"

struct TemplateStruct {
  int a;
  int b;

  [[nodiscard]] std::vector<int> to_vector() const {
    return std::vector<int>{a, b};
  }
};

TEST_CASE("Check to_vector concept") {

  const TemplateStruct test_object{1, 0};

  const auto vectorized_object =
      cpp_template::util::to_vector<int, TemplateStruct>(test_object);

  CHECK(vectorized_object.size() == 2);
  CHECK(vectorized_object[0] == 1);
  CHECK(vectorized_object[1] == 0);
}