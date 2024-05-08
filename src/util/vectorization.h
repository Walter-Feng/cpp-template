#ifndef CPP_TEMPLATE_UTIL_VECTORIZATION_H
#define CPP_TEMPLATE_UTIL_VECTORIZATION_H


#include <iostream>
#include <vector>

namespace cpp_template::util {

/// \brief a concept of objects that have [...] member function that
/// works like an array / a vector
template<typename T>
concept Indexable = requires(T object) {
  { object[size_t()] };
};

/// \brief a concept of objects that have .to_vector() member function that
/// outputs an indexable object
template<typename T>
concept Vectorizable = requires(T object) {
  { object.to_vector() } -> Indexable;
};

/// \brief A wrapper of to_vector for objects that have member function
/// \param object the object to be vectorised
/// \return indexable object
Indexable auto to_vector(const Vectorizable auto & object) {
  return object.to_vector();
}

}

#endif //CPP_TEMPLATE_UTIL_VECTORIZATION_H
