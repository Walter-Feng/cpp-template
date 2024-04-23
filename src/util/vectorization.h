#ifndef CPPTEMPLATE_UTIL_VECTORIZATION_H
#define CPPTEMPLATE_UTIL_VECTORIZATION_H


#include <iostream>
#include <vector>

namespace cpp_template::util {

template<typename T>
concept Indexable = requires(T object) {
  { object[size_t()] };
};

template<class Object>
concept Vectorizable = requires(Object object) {
  { object.to_vector() } -> Indexable;
};

Indexable auto to_vector(const Vectorizable auto & object) {
  return object.to_vector();
}

}

#endif //CPPTEMPLATE_UTIL_VECTORIZATION_H
