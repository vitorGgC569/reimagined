#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "include/Tensor.h"
#include "include/TensorOps.h"
#include "include/LoRA.h"
#include "include/DoRA.h"
#include "include/IA3.h"
#include "include/TurboFusion.h"
#include "include/FullFinetuning.h"

// --- Dataset Generation ---
struct Dataset {
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
};

Dataset generate_dataset(int num_samples, int input_dim, int output_dim) {
    Dataset data;
    std::default_random_engine generator(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // True weights for the synthetic task
    Tensor true_weights(output_dim, input_dim);
    for(int i=0; i<output_dim; ++i)
        for(int j=0; j<input_dim; ++j)
            true_weights.at(i, j) = dist(generator);

    for (int i = 0; i < num_samples; ++i) {
        Tensor x(1, input_dim);
        for(int j=0; j<input_dim; ++j) x.at(0, j) = dist(generator);

        // y = x * W^T + noise
        Tensor y = TensorOps::multiply(x, true_weights.transpose());
        // Add minimal noise
        for(int j=0; j<output_dim; ++j) y.at(0, j) += dist(generator) * 0.01f;

        data.inputs.push_back(x);
        data.targets.push_back(y);
    }
    return data;
}

// --- Training Loop Template ---
template <typename LayerType>
void train_model(std::string name, LayerType& layer, const Dataset& train_data, int epochs, float lr) {
    std::cout << "Training " << name << "..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    float final_loss = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < train_data.inputs.size(); ++i) {
            // Forward
            Tensor pred = layer.forward(train_data.inputs[i]);

            // Loss (MSE)
            // L = 0.5 * (pred - target)^2
            // dL/dPred = pred - target

            Tensor diff(pred.getRows(), pred.getCols());
            float sample_loss = 0.0f;
            for(int r=0; r<pred.getRows(); ++r) {
                for(int c=0; c<pred.getCols(); ++c) {
                    float d = pred.at(r, c) - train_data.targets[i].at(r, c);
                    diff.at(r, c) = d;
                    sample_loss += 0.5f * d * d;
                }
            }
            epoch_loss += sample_loss;

            // Backward
            layer.backward(diff);

            // Update
            layer.update(lr);
        }

        if (epoch == epochs - 1) final_loss = epoch_loss / train_data.inputs.size();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "Result [" << name << "]: Final Loss = " << final_loss
              << ", Time = " << duration.count() << "s" << std::endl;
}

int main() {
    int input_dim = 64;
    int output_dim = 64;
    int rank = 8;
    int num_samples = 200;
    int epochs = 10;
    float lr = 0.01f;

    std::cout << "--- PEFT Comparative Benchmark ---" << std::endl;
    std::cout << "Task: Multivariate Regression (Synthetic)" << std::endl;
    std::cout << "Samples: " << num_samples << ", Epochs: " << epochs << ", Rank: " << rank << std::endl;
    std::cout << "Input Dim: " << input_dim << ", Output Dim: " << output_dim << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    Dataset data = generate_dataset(num_samples, input_dim, output_dim);

    // Shared Base Weights (Random Initialization)
    Tensor base_weights(output_dim, input_dim);
    std::default_random_engine generator(99);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for(int i=0; i<output_dim; ++i)
        for(int j=0; j<input_dim; ++j)
            base_weights.at(i, j) = dist(generator);

    // 1. LoRA
    {
        LoRALayer model(input_dim, output_dim, rank);
        model.setBaseWeights(base_weights);
        train_model("LoRA", model, data, epochs, lr);
    }

    // 2. DoRA
    {
        DoRALayer model(input_dim, output_dim, rank);
        model.setBaseWeights(base_weights);
        train_model("DoRA", model, data, epochs, lr);
    }

    // 3. IA3
    {
        IA3Layer model(input_dim, output_dim);
        model.setBaseWeights(base_weights);
        // IA3 often needs smaller LR or different tuning, but we compare equal footing
        train_model("IA3 ", model, data, epochs, lr);
    }

    // 4. TurboFusion
    {
        TurboFusionLayer model(input_dim, output_dim, rank);
        model.setBaseWeights(base_weights);
        train_model("TurboFusion", model, data, epochs, lr);
    }

    return 0;
}
