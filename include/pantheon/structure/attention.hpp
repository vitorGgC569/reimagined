#pragma once

#include <vector>
#include <cmath>
#include <numeric>

namespace pantheon {
namespace structure {

    class AttentionTransfer {
    public:
        // CAT-KD: Class Attention Transfer
        // Computes CAM and MSE loss
        // weights: (NumClasses, Channels) - FC layer weights
        // features: (Batch, Channels, Height, Width)
        static float compute_cam_loss(const std::vector<float>& s_feat, const std::vector<float>& s_weights,
                                    const std::vector<float>& t_feat, const std::vector<float>& t_weights,
                                    int batch, int channels, int height, int width, int num_classes, int target_class) {

            auto s_cam = compute_cam(s_feat, s_weights, batch, channels, height, width, num_classes, target_class);
            auto t_cam = compute_cam(t_feat, t_weights, batch, channels, height, width, num_classes, target_class);

            // Normalize CAMs
            normalize(s_cam);
            normalize(t_cam);

            // MSE Loss
            float loss = 0.0f;
            for (size_t i = 0; i < s_cam.size(); ++i) {
                float diff = s_cam[i] - t_cam[i];
                loss += diff * diff;
            }
            return loss / s_cam.size();
        }

    private:
        static std::vector<float> compute_cam(const std::vector<float>& feat, const std::vector<float>& weights,
                                            int B, int C, int H, int W, int num_classes, int target_class) {

            std::vector<float> cam(B * H * W, 0.0f);

            // CAM = Sum_k (w_k * f_k)
            // w_k = weights[target_class][k]

            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        float val = 0.0f;
                        for (int c = 0; c < C; ++c) {
                            float weight = weights[target_class * C + c];
                            float f_val = feat[b * (C*H*W) + c * (H*W) + h * W + w];
                            val += weight * f_val;
                        }
                        cam[b * (H*W) + h * W + w] = val;
                    }
                }
            }
            return cam;
        }

        static void normalize(std::vector<float>& v) {
            float min_val = 1e9f;
            float max_val = -1e9f;
            for (float f : v) {
                if (f < min_val) min_val = f;
                if (f > max_val) max_val = f;
            }
            float range = max_val - min_val;
            if (range < 1e-6f) range = 1.0f;

            for (float& f : v) {
                f = (f - min_val) / range;
            }
        }
    };

}
}
