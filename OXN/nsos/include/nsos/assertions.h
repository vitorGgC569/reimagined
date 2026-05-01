#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>


// Pre-declare Inspector for safety check integration
namespace nsos {
class Inspector;
}

#ifdef NSOS_ENABLE_ASSERTS
#include "nsos/inspector.h"
#define NSOS_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      nsos::Inspector::instance().panic("Assertion failed: " #cond);           \
    }                                                                          \
  } while (0)

#define NSOS_ASSERT_EQ(a, b) NSOS_ASSERT((a) == (b))
#define NSOS_ASSERT_NEAR(a, b, tol) NSOS_ASSERT(std::abs((a) - (b)) < (tol))

#define NSOS_ASSERT_TENSOR_SHAPE(t, ...)                                       \
  do {                                                                         \
    auto expected = std::vector<int>{__VA_ARGS__};                             \
    NSOS_ASSERT(t.shape == expected);                                          \
  } while (0)

#define NSOS_ASSERT_TENSOR_FINITE(t)                                           \
  do {                                                                         \
    NSOS_ASSERT(!t.has_nan());                                                 \
    NSOS_ASSERT(!t.has_inf());                                                 \
  } while (0)
#else
#define NSOS_ASSERT(cond) ((void)0)
#define NSOS_ASSERT_EQ(a, b) ((void)0)
#define NSOS_ASSERT_NEAR(a, b, tol) ((void)0)
#define NSOS_ASSERT_TENSOR_SHAPE(t, ...) ((void)0)
#define NSOS_ASSERT_TENSOR_FINITE(t) ((void)0)
#endif
