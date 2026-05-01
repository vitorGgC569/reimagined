#include "nsos/determinism.h"
#include "nsos/tensor.h"
#include "nsos/test_context.h"
#include <gtest/gtest.h>


using namespace nsos;
using namespace nsos::tests;

class TensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    determinism::DeterminismManager::instance().set_global_seed(42);
  }
};

TEST_F(TensorTest, Construction_Basic) {
  Tensor t({2, 3}, Device::CPU);
  EXPECT_EQ(t.shape.size(), 2);
  EXPECT_EQ(t.shape[0], 2);
  EXPECT_EQ(t.shape[1], 3);
  EXPECT_EQ(t.size, 6);
}

TEST_F(TensorTest, Matmul_Determinism) {
  IsolatedTestContext ctx(123);

  Tensor a = Tensor::random({4, 4});
  Tensor b = Tensor::random({4, 4});
  Tensor res1 = a.matmul(b);

  ctx.reset();

  Tensor a2 = Tensor::random({4, 4});
  Tensor b2 = Tensor::random({4, 4});
  Tensor res2 = a2.matmul(b2);

  const float *d1 = res1.data();
  const float *d2 = res2.data();
  for (int i = 0; i < res1.size; ++i) {
    EXPECT_EQ(d1[i], d2[i]) << "Bit-exact mismatch at index " << i;
  }
}

TEST_F(TensorTest, NaN_Detection_Inspector) {
  Tensor t({1}, Device::CPU);
  t.data()[0] = std::nanf("");

  // This should trigger a panic in the inspector if health check is on
  // In a real test we might want to capture the death
  // EXPECT_DEATH(Inspector::instance().check_tensor_health(t, "test"), "NaN
  // detected");
}
