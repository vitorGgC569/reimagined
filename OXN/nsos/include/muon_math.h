#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nsos::muon {
inline constexpr const char* kIdentity = "muon_hidden_ns5_fp32_fp64norm_nesterov095_v1";
inline constexpr const char* kReferenceCommit = "f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd";
inline constexpr const char* kReferenceSha256 = "1eece0b562f159e88f3a65da9b6ee8d0ab2dd647cf25c88e355d4f262e796e66";
inline constexpr float kMomentum = 0.95f;
inline constexpr float kA = 3.4445f, kB = -4.7750f, kC = 2.0315f;
inline constexpr float kEpsilon = 1e-7f;
inline constexpr int kSteps = 5;

// Registry identity is stable. Muon is restricted to hidden matrix weights;
// embeddings/output/tied heads/router and all scalar/vector parameters retain
// AdamW. Explicit no-decay classification remains the caller's authority.
inline bool hidden_matrix(const std::string& name, const std::vector<int>& shape) {
    return shape.size() == 2 && shape[0] > 0 && shape[1] > 0 &&
        name.rfind("layers.", 0) == 0 && name.size() >= 7 &&
        name.compare(name.size()-7, 7, ".weight") == 0 &&
        name.find(".router.") == std::string::npos &&
        name.find("embedding") == std::string::npos &&
        name.find("lm_head") == std::string::npos;
}

// Independent CPU reference for the FP32 adaptation of pinned KellerJordan
// NS5. This is a quintic approximate polar transform, NOT an exact SVD polar
// factor. Upstream rounds its iteration to BF16; this policy deliberately keeps
// FP32 and records that difference in the runtime identity.
inline std::vector<float> direction(const std::vector<float>& gradient,
    std::vector<float>& momentum, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || size_t(rows) > std::numeric_limits<size_t>::max()/size_t(cols) ||
        gradient.size() != size_t(rows)*size_t(cols) || momentum.size() != gradient.size())
        throw std::invalid_argument("Muon matrix shape mismatch");
    const bool transposed = rows > cols;
    const int r = std::min(rows, cols), c = std::max(rows, cols);
    auto next = momentum;
    std::vector<float> x(gradient.size()), y(x.size()), gram(size_t(r)*r), poly(gram.size());
    double norm2 = 0;
    for (size_t i=0; i<gradient.size(); ++i) {
        if (!std::isfinite(gradient[i]) || !std::isfinite(momentum[i]))
            throw std::invalid_argument("Muon nonfinite input");
        next[i] = momentum[i]*kMomentum + gradient[i]*(1-kMomentum);
        const float u = gradient[i]*(1-kMomentum) + next[i]*kMomentum;
        const int row = int(i/cols), col = int(i%cols);
        x[transposed ? size_t(col)*c+row : i] = u;
        norm2 += double(u)*u;
    }
    const float denominator = float(std::sqrt(norm2)) + kEpsilon;
    if(!std::isfinite(denominator))throw std::overflow_error("Muon normalization overflow");
    for (auto& v : x) v /= denominator;
    for (int iteration=0; iteration<kSteps; ++iteration) {
        for (int i=0;i<r;++i) for (int j=0;j<r;++j) {
            float sum=0; for(int k=0;k<c;++k) sum += x[size_t(i)*c+k]*x[size_t(j)*c+k];
            gram[size_t(i)*r+j]=sum;
        }
        for (int i=0;i<r;++i) for (int j=0;j<r;++j) {
            float sum=0; for(int k=0;k<r;++k) sum += gram[size_t(i)*r+k]*gram[size_t(k)*r+j];
            poly[size_t(i)*r+j]=kB*gram[size_t(i)*r+j]+kC*sum;
        }
        for (int i=0;i<r;++i) for (int j=0;j<c;++j) {
            float sum=0; for(int k=0;k<r;++k) sum += poly[size_t(i)*r+k]*x[size_t(k)*c+j];
            y[size_t(i)*c+j]=kA*x[size_t(i)*c+j]+sum;
        }
        x.swap(y);
    }
    const float adjustment = std::sqrt(std::max(1.0f, float(rows)/cols));
    std::vector<float> result(gradient.size());
    for(int i=0;i<rows;++i) for(int j=0;j<cols;++j) {
        const float value = x[transposed ? size_t(j)*c+i : size_t(i)*c+j]*adjustment;
        if(!std::isfinite(value) || !std::isfinite(next[size_t(i)*cols+j]))
            throw std::overflow_error("Muon direction overflow");
        result[size_t(i)*cols+j]=value;
    }
    momentum.swap(next); // Publish only after the entire finite direction exists.
    return result;
}
} // namespace nsos::muon
