#ifndef CPP_TEMPLATE_UTIL_H
#define CPP_TEMPLATE_UTIL_H

#include <iostream>
#include <vector>

namespace cpp_template::util {

template<typename T>
concept Indexable = requires(T object) {
  {object[size_t()]};
};

template<class Object>
concept Vectorizable = requires(Object object) {
  {object.to_vector()} -> Indexable;
};

template<class Object>
Indexable auto to_vector(const Vectorizable auto & object) {
  return object.to_vector();
}

}

#endif //CPP_TEMPLATE_UTIL_H
