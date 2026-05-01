#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <iomanip>
#include <random>

#include "include/Tensor.h"
#include "include/TensorOps.h"
#include "include/LoRA.h"
#include "include/FullFinetuning.h"
#include "include/IA3.h"
#include "include/DoRA.h"
#include "include/TurboFusion.h"

// --- Helper Functions to Replace Missing Methods ---
void tensor_fill(Tensor& t, float val) {
    for (int i = 0; i < t.getRows(); ++i) {
        for (int j = 0; j < t.getCols(); ++j) {
            t.at(i, j) = val;
        }
    }
}

void tensor_randomize(Tensor& t) {
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution(-1.0, 1.0);
    for (int i = 0; i < t.getRows(); ++i) {
        for (int j = 0; j < t.getCols(); ++j) {
            t.at(i, j) = distribution(generator);
        }
    }
}

void tensor_zeros(Tensor& t) {
    tensor_fill(t, 0.0f);
}

// --- Simple Test Framework ---
namespace TestFramework {
    int tests_run = 0;
    int tests_failed = 0;

    void run_test(const std::string& name, std::function<void()> test_func) {
        tests_run++;
        std::cout << "[RUN] " << name << " ... ";
        try {
            test_func();
            std::cout << "PASS" << std::endl;
        } catch (const std::exception& e) {
            tests_failed++;
            std::cout << "FAIL: " << e.what() << std::endl;
        } catch (...) {
            tests_failed++;
            std::cout << "FAIL: Unknown error" << std::endl;
        }
    }

    void print_summary() {
        std::cout << "\n--- Test Summary ---" << std::endl;
        std::cout << "Tests Run: " << tests_run << std::endl;
        std::cout << "Passed: " << (tests_run - tests_failed) << std::endl;
        std::cout << "Failed: " << tests_failed << std::endl;
    }

    void assert_true(bool condition, const std::string& message) {
        if (!condition) throw std::runtime_error(message);
    }

    void assert_close(float a, float b, float tol = 1e-4) {
        if (std::abs(a - b) > tol) {
            throw std::runtime_error("Values not close: " + std::to_string(a) + " vs " + std::to_string(b));
        }
    }
}

using namespace TestFramework;

// --- Tests ---

void test_tensor_ops() {
    Tensor A(2, 3);
    tensor_fill(A, 1.0f);
    Tensor B(3, 2);
    tensor_fill(B, 2.0f);

    // Test Matmul
    // [1 1 1] * [2 2] = [6 6]
    // [1 1 1]   [2 2]   [6 6]
    //           [2 2]
    Tensor C = TensorOps::multiply(A, B);
    assert_true(C.getRows() == 2 && C.getCols() == 2, "Matmul dims incorrect");
    assert_close(C.at(0, 0), 6.0f);
    assert_close(C.at(1, 1), 6.0f);
}

void test_full_finetuning() {
    FullFinetuningLayer layer(2, 2);
    Tensor input(1, 2);
    input.at(0, 0) = 1.0f; input.at(0, 1) = 2.0f;

    Tensor output = layer.forward(input);
    assert_true(output.getRows() == 1 && output.getCols() == 2, "Output dims incorrect");

    Tensor grad(1, 2);
    tensor_fill(grad, 0.1f);
    layer.backward(grad);

    // Check if gradients were generated (not null)
    assert_true(layer.get_grad_W() != nullptr, "Grad W is null");
    assert_true(layer.get_grad_B() != nullptr, "Grad B is null");
}

void test_ia3() {
    IA3Layer layer(2, 2);
    Tensor W0(2, 2);
    tensor_fill(W0, 1.0f); // Identity-like effect if we ignore dims, but here all 1s.
    layer.setBaseWeights(W0);

    Tensor input(1, 2);
    tensor_fill(input, 1.0f);

    // Y_pre = [1 1] * [1 1; 1 1]^T = [1 1] * [1 1; 1 1] = [2 2]
    // IA3 scales by L (initially 1). Result should be [2 2].
    Tensor output = layer.forward(input);
    assert_close(output.at(0, 0), 2.0f);

    Tensor grad(1, 2);
    tensor_fill(grad, 1.0f);
    layer.backward(grad);

    // dL/dL = grad * Y_pre = 1 * 2 = 2
    assert_true(layer.get_grad_L() != nullptr, "Grad L is null");
    assert_close(layer.get_grad_L()->at(0, 0), 2.0f);
}

void test_dora() {
    DoRALayer layer(2, 2, 2); // rank 2
    Tensor W0(2, 2);
    tensor_fill(W0, 1.0f); // Norm of each row = sqrt(1+1) = 1.414
    layer.setBaseWeights(W0);

    Tensor input(1, 2);
    tensor_fill(input, 1.0f);

    // Init: m = norm(W0). B=0.
    // V = W0. V_bar = W0 / norm(W0).
    // W' = m * V_bar = m * W0 / m = W0.
    // So initially, DoRA should behave exactly like W0.
    // Output = [1 1] * [1 1; 1 1]^T = [2 2]

    Tensor output = layer.forward(input);
    assert_close(output.at(0, 0), 2.0f);

    Tensor grad(1, 2);
    tensor_fill(grad, 0.1f);
    layer.backward(grad);

    assert_true(layer.get_grad_m() != nullptr, "Grad m is null");
}

void test_turbo_fusion() {
    TurboFusionLayer layer(4, 4, 2);
    Tensor W0(4, 4);
    tensor_randomize(W0);
    layer.setBaseWeights(W0);

    Tensor input(1, 4);
    tensor_randomize(input);

    Tensor output = layer.forward(input);
    assert_true(output.getCols() == 4, "Output dim mismatch");

    Tensor grad(1, 4);
    tensor_randomize(grad);
    layer.backward(grad);

    // Check if underlying DoRA updated (indirectly checking if it didn't crash)
    assert_true(layer.getDoRA().get_grad_m() != nullptr, "Underlying DoRA grad missing");
}

int main() {
    run_test("Tensor Operations", test_tensor_ops);
    run_test("Full Finetuning", test_full_finetuning);
    run_test("IA3 Forward/Backward", test_ia3);
    run_test("DoRA Initialization", test_dora);
    run_test("TurboFusion Wrapper", test_turbo_fusion);

    print_summary();
    return tests_failed > 0 ? 1 : 0;
}
