#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include "contrastive.hpp"

namespace pantheon {
namespace structure {

    class MultiScaleDistillation {
    public:
        // MSDCRD: Multi-Scale Decoupling Contrastive Representation Distillation
        // Decouples feature map into patches and applies CRD
        // feat: (Batch, Channels, Height, Width) flattened
        static float compute_msdcrd_loss(const std::vector<float>& s_feat,
                                       const std::vector<float>& t_feat,
                                       int batch, int channels, int height, int width,
                                       int patch_size_h, int patch_size_w) {

            // 1. Pooling/Slicing
            // We implement sliding window (or grid) decoupling.
            // For simplicity, we implement grid patches (non-overlapping).

            std::vector<float> s_patches;
            std::vector<float> t_patches;

            // Extract patches
            // Loop over batch
            // Loop over grid
            // Flatten patch into vector

            // Grid dimensions
            int grid_h = height / patch_size_h;
            int grid_w = width / patch_size_w;

            if (grid_h * patch_size_h != height || grid_w * patch_size_w != width) {
                // In real impl, handle padding. For now, strict.
                // We'll proceed with valid grid area.
            }

            int patch_dim = channels * patch_size_h * patch_size_w;

            // We treat each patch as a "sample" for CRD
            // Total samples = batch * grid_h * grid_w

            for (int b = 0; b < batch; ++b) {
                for (int gh = 0; gh < grid_h; ++gh) {
                    for (int gw = 0; gw < grid_w; ++gw) {
                        // Extract patch (gh, gw) for item b
                        extract_patch(s_feat, s_patches, b, gh, gw,
                                    channels, height, width, patch_size_h, patch_size_w);
                        extract_patch(t_feat, t_patches, b, gh, gw,
                                    channels, height, width, patch_size_h, patch_size_w);
                    }
                }
            }

            int total_samples = batch * grid_h * grid_w;
            return ContrastiveDistillation::compute_loss(s_patches, t_patches, total_samples, patch_dim);
        }

    private:
        static void extract_patch(const std::vector<float>& src, std::vector<float>& dest,
                                int b, int gh, int gw,
                                int C, int H, int W, int PH, int PW) {

            int start_h = gh * PH;
            int start_w = gw * PW;

            // NCHW layout assumption
            // Offset = b * (C*H*W) + c * (H*W) + h * W + w

            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < PH; ++h) {
                    for (int w = 0; w < PW; ++w) {
                        int src_idx = b * (C*H*W) + c * (H*W) + (start_h + h) * W + (start_w + w);
                        dest.push_back(src[src_idx]);
                    }
                }
            }
        }
    };

}
}
