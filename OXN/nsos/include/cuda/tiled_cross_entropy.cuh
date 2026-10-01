#pragma once
#ifdef USE_CUDA
extern "C" {
void launch_cce_mask(float*, const float*, int, int);
void launch_cce_statistics(const float*, float*, const float*, int, int, int);
void launch_cce_finish(float*, float*, const float*, int, int, float, float);
void launch_cce_gradient(const float*, float*, const float*, const float*,
                         int, int, int, int, float, float, float);
void launch_cce_add_region(float*, const float*, int, int, int, int, int);
}
#endif
