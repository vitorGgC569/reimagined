#include "optimizer_4bit.h"

#include <algorithm>
#include <cmath>

namespace nsos {

namespace {

// Signed dynamic-exponent-style code map for the first moment.  16 levels,
// symmetric, exponentially spaced: ±{2^0, 2^-1, ..., 2^-7}.  Resolution is
// concentrated near zero (where normalised momentum mass sits) exactly as the
// paper's dynamic-exponent map intends; the largest magnitude is 1.0 so a value
// equal to the block abs-max is represented exactly.
const float kFirstMomentMap[kQuant4Levels] = {
    -1.0f,       -0.5f,        -0.25f,      -0.125f,
    -0.0625f,    -0.03125f,    -0.015625f,  -0.0078125f,
     0.0078125f,  0.015625f,    0.03125f,    0.0625f,
     0.125f,      0.25f,        0.5f,        1.0f};

inline void set_nibble(std::vector<uint8_t>& buf, int i, uint8_t code) {
    uint8_t& b = buf[static_cast<size_t>(i) >> 1];
    if (i & 1) {
        b = static_cast<uint8_t>((b & 0x0F) | static_cast<uint8_t>((code & 0x0F) << 4));
    } else {
        b = static_cast<uint8_t>((b & 0xF0) | (code & 0x0F));
    }
}

inline uint8_t get_nibble(const std::vector<uint8_t>& buf, int i) {
    const uint8_t b = buf[static_cast<size_t>(i) >> 1];
    return (i & 1) ? static_cast<uint8_t>(b >> 4) : static_cast<uint8_t>(b & 0x0F);
}

// Nearest signed code for a value already normalised to [-1, 1].
inline uint8_t quantize_first_moment(float normalized) {
    uint8_t best = 0;
    float best_err = std::abs(kFirstMomentMap[0] - normalized);
    for (int k = 1; k < kQuant4Levels; ++k) {
        const float err = std::abs(kFirstMomentMap[k] - normalized);
        if (err < best_err) {
            best_err = err;
            best = static_cast<uint8_t>(k);
        }
    }
    return best;
}

// Linear no-zero map for the second moment: a normalised value x01 in [0, 1]
// maps to the nearest level (k+1)/16.  Inverse of T(k) = (k+1)/16.
inline uint8_t quantize_second_moment(float x01) {
    x01 = std::min(std::max(x01, 0.0f), 1.0f);
    int k = static_cast<int>(std::lround(x01 * static_cast<float>(kQuant4Levels) - 1.0f));
    k = std::min(std::max(k, 0), kQuant4Levels - 1);
    return static_cast<uint8_t>(k);
}

inline float dequantize_second_moment(uint8_t code) {
    return static_cast<float>(code + 1) / static_cast<float>(kQuant4Levels);
}

}  // namespace

std::size_t Quant4OptState::bytes() const {
    if (!quantized) {
        return (m_fp32.size() + v_fp32.size()) * sizeof(float);
    }
    std::size_t total = m_codes.size() + v_codes.size();
    total += m_absmax.size() * sizeof(float);
    if (v_rank1) {
        total += (v_row.size() + v_col.size()) * sizeof(float);
    } else {
        total += v_absmax.size() * sizeof(float);
    }
    return total;
}

void quant4_store_m(const float* m, int n, Quant4OptState& st) {
    st.n = n;
    if (n <= kQuant4MinElems) {
        st.quantized = false;
        st.m_fp32.assign(m, m + n);
        return;
    }
    st.quantized = true;
    const int num_blocks =
        n / kQuant4BlockSize + static_cast<int>(n % kQuant4BlockSize != 0);
    st.m_absmax.assign(static_cast<size_t>(num_blocks), 0.0f);
    st.m_codes.assign(
        static_cast<size_t>(n / 2 + static_cast<int>(n % 2 != 0)), 0);

    for (int blk = 0; blk < num_blocks; ++blk) {
        const int start = blk * kQuant4BlockSize;
        const int end = std::min(start + kQuant4BlockSize, n);
        float absmax = 0.0f;
        for (int i = start; i < end; ++i) {
            absmax = std::max(absmax, std::abs(m[i]));
        }
        st.m_absmax[static_cast<size_t>(blk)] = absmax;
        const float inv = absmax > 0.0f ? 1.0f / absmax : 0.0f;
        for (int i = start; i < end; ++i) {
            const uint8_t code = inv > 0.0f ? quantize_first_moment(m[i] * inv)
                                            : static_cast<uint8_t>(0);
            set_nibble(st.m_codes, i, code);
        }
    }
}

void quant4_load_m(const Quant4OptState& st, float* m_out, int n) {
    if (!st.quantized) {
        std::copy(st.m_fp32.begin(), st.m_fp32.begin() + n, m_out);
        return;
    }
    const int num_blocks =
        n / kQuant4BlockSize + static_cast<int>(n % kQuant4BlockSize != 0);
    for (int blk = 0; blk < num_blocks; ++blk) {
        const int start = blk * kQuant4BlockSize;
        const int end = std::min(start + kQuant4BlockSize, n);
        const float absmax = st.m_absmax[static_cast<size_t>(blk)];
        for (int i = start; i < end; ++i) {
            m_out[i] = kFirstMomentMap[get_nibble(st.m_codes, i)] * absmax;
        }
    }
}

void quant4_store_v(const float* v, int n, int rows, int cols, Quant4OptState& st) {
    st.n = n;
    if (n <= kQuant4MinElems) {
        st.quantized = false;
        st.v_fp32.assign(v, v + n);
        return;
    }
    st.quantized = true;
    st.v_codes.assign(
        static_cast<size_t>(n / 2 + static_cast<int>(n % 2 != 0)), 0);

    const bool use_rank1 = rows > 0 && cols > 0 &&
                           static_cast<long long>(rows) * cols == n;
    st.v_rank1 = use_rank1;

    if (use_rank1) {
        st.v_rows = rows;
        st.v_cols = cols;
        st.v_row.assign(static_cast<size_t>(rows), 0.0f);
        st.v_col.assign(static_cast<size_t>(cols), 0.0f);
        // r_i = max_j x_ij,  c_j = max_i x_ij  (v is non-negative).
        for (int i = 0; i < rows; ++i) {
            float row_max = 0.0f;
            const float* vi = v + static_cast<size_t>(i) * cols;
            for (int j = 0; j < cols; ++j) {
                const float val = vi[j];
                row_max = std::max(row_max, val);
                if (val > st.v_col[static_cast<size_t>(j)]) {
                    st.v_col[static_cast<size_t>(j)] = val;
                }
            }
            st.v_row[static_cast<size_t>(i)] = row_max;
        }
        for (int i = 0; i < rows; ++i) {
            const float* vi = v + static_cast<size_t>(i) * cols;
            const float ri = st.v_row[static_cast<size_t>(i)];
            for (int j = 0; j < cols; ++j) {
                const float s = std::min(ri, st.v_col[static_cast<size_t>(j)]);
                const float x01 = s > 0.0f ? vi[j] / s : 0.0f;
                set_nibble(st.v_codes, i * cols + j, quantize_second_moment(x01));
            }
        }
        return;
    }

    // Block-wise abs-max fallback (still zero-safe via the linear no-zero map).
    const int num_blocks =
        n / kQuant4BlockSize + static_cast<int>(n % kQuant4BlockSize != 0);
    st.v_absmax.assign(static_cast<size_t>(num_blocks), 0.0f);
    for (int blk = 0; blk < num_blocks; ++blk) {
        const int start = blk * kQuant4BlockSize;
        const int end = std::min(start + kQuant4BlockSize, n);
        float absmax = 0.0f;
        for (int i = start; i < end; ++i) {
            absmax = std::max(absmax, v[i]);  // v >= 0
        }
        st.v_absmax[static_cast<size_t>(blk)] = absmax;
        const float inv = absmax > 0.0f ? 1.0f / absmax : 0.0f;
        for (int i = start; i < end; ++i) {
            const float x01 = inv > 0.0f ? v[i] * inv : 0.0f;
            set_nibble(st.v_codes, i, quantize_second_moment(x01));
        }
    }
}

void quant4_load_v(const Quant4OptState& st, float* v_out, int n) {
    if (!st.quantized) {
        std::copy(st.v_fp32.begin(), st.v_fp32.begin() + n, v_out);
        return;
    }
    if (st.v_rank1) {
        const int rows = st.v_rows;
        const int cols = st.v_cols;
        for (int i = 0; i < rows; ++i) {
            const float ri = st.v_row[static_cast<size_t>(i)];
            float* vo = v_out + static_cast<size_t>(i) * cols;
            for (int j = 0; j < cols; ++j) {
                const float s = std::min(ri, st.v_col[static_cast<size_t>(j)]);
                vo[j] = dequantize_second_moment(get_nibble(st.v_codes, i * cols + j)) * s;
            }
        }
        return;
    }
    const int num_blocks =
        n / kQuant4BlockSize + static_cast<int>(n % kQuant4BlockSize != 0);
    for (int blk = 0; blk < num_blocks; ++blk) {
        const int start = blk * kQuant4BlockSize;
        const int end = std::min(start + kQuant4BlockSize, n);
        const float absmax = st.v_absmax[static_cast<size_t>(blk)];
        for (int i = start; i < end; ++i) {
            v_out[i] = dequantize_second_moment(get_nibble(st.v_codes, i)) * absmax;
        }
    }
}

}  // namespace nsos
