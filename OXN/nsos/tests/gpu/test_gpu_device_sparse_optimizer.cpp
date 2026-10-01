#include "gpu_parity_common.h"
#include "cuda/sparse_optimizer_activity.cuh"

#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

#ifdef USE_CUDA
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Native device allocations, never managed-memory host dereferences.
template <class T> struct DeviceValues {
    T* data = nullptr;
    size_t count;
    explicit DeviceValues(size_t n) : count(n) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(T)));
    }
    ~DeviceValues() { if (data) (void)cudaFree(data); }
    DeviceValues(const DeviceValues&) = delete;
    DeviceValues& operator=(const DeviceValues&) = delete;
    void put(const std::vector<T>& values) {
        require(values.size() == count, "fixture upload size mismatch");
        check(cudaStreamSynchronize(nsos::gpu::current_stream()));
        check(cudaMemcpy(data, values.data(), count * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> get() const {
        std::vector<T> result(count);
        check(cudaStreamSynchronize(nsos::gpu::current_stream()));
        check(cudaMemcpy(result.data(), data, count * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }
};

template <class T> void equal(const std::vector<T>& actual,
    const std::vector<T>& expected, const char* message) {
    require(actual.size() == expected.size(), message);
    require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0, message);
}

void close(float actual, float expected, const char* message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= 2e-6f, message);
}

struct Fixture {
    DeviceValues<unsigned char> accumulated{3}, current{3}, first{3}, initialized{3};
    DeviceValues<int> abort_issue{1}, finite_issue{1}, output_issue{1}, offsets{4}, upstream_issue{1};
    DeviceValues<float> w{6}, g{6}, m{6}, v{6}, lr{3}, sqsum{1}, coefficient{1};
    DeviceValues<double> total{1}, partials{3};
    DeviceValues<float*> weights{3}, gradients{3}, moments1{3}, moments2{3};
    DeviceValues<const float*> values{3};
    DeviceValues<unsigned long long> tensor_offsets{4};
    DeviceValues<unsigned char> wd{3}, nonnegative{3};
    DeviceValues<const unsigned char*> predicates{3};
    DeviceValues<NsosMultiTensorChunk> chunks{3};
    NsosSparseOptimizerActivity activity;
    NsosActivityAwareOptimizerDesc desc{};

    Fixture() : activity{accumulated.data, current.data, first.data, abort_issue.data, 3} {
        weights.put({w.data, w.data + 2, w.data + 4});
        gradients.put({g.data, g.data + 2, g.data + 4});
        moments1.put({m.data, m.data + 2, m.data + 4});
        moments2.put({v.data, v.data + 2, v.data + 4});
        values.put({g.data, g.data + 2, g.data + 4});
        predicates.put({accumulated.data, accumulated.data + 1, accumulated.data + 2});
        tensor_offsets.put({0, 2, 4, 6});
        chunks.put({{0, 2, 0}, {0, 2, 1}, {0, 2, 2}});
        wd.put({1, 1, 1});
        nonnegative.put({1, 1, 1});
        lr.put({0.1f, 0.1f, 0.1f});
        w.put({1, 2, 3, 4, 5, 6});
        g.put({3, 4, 0, 0, 0, 0});
        m.put({0, 0, 0, 0, 0, 0});
        v.put({0, 0, 0, 0, 0, 0});
        initialized.put({0, 0, 0});
        upstream_issue.put({0});
        desc.w = weights.data; desc.g = gradients.data;
        desc.m = moments1.data; desc.v = moments2.data;
        desc.offsets = tensor_offsets.data; desc.wd_flags = wd.data;
        desc.learning_rates = lr.data; desc.n_tensors = 3; desc.total = 6;
        desc.contribution = {predicates.data, abort_issue.data};
        reset();
    }
    void reset() {
        require(launch_sparse_optimizer_activity_reset(activity), "activity reset enqueue failed");
        finite_issue.put({0}); output_issue.put({0}); sqsum.put({0});
        desc.contribution.abort_issue = abort_issue.data;
    }
    void route(const std::vector<int>& expert_offsets, int capacity,
               const int* issue = nullptr) {
        offsets.put(expert_offsets);
        require(launch_sparse_optimizer_activity_update(activity, offsets.data, capacity, issue),
                "activity update enqueue failed");
    }
    void finite(bool chunked) {
        require(launch_activity_multi_tensor_check_finite(finite_issue.data, values.data,
                    nullptr, desc, chunked ? chunks.data : nullptr, chunked ? 3 : 0),
                "finite scan enqueue failed");
    }
    void adam(bool chunked) {
        require(launch_activity_multi_tensor_adamw_update_deterministic(desc,
                    chunked ? chunks.data : nullptr, chunked ? 3 : 0,
                    0.5f, 0.5f, 0.5f, 0.5f, 1e-8f, 0.2f, output_issue.data),
                "deterministic Adam enqueue failed");
    }
};

void accumulation_and_invalid_routing() {
    Fixture f;
    f.route({0, 2, 2, 2}, 2);
    equal<unsigned char>(f.accumulated.get(), {1, 0, 0}, "first union differs");
    equal<unsigned char>(f.first.get(), {1, 0, 0}, "first write differs");
    f.route({0, 0, 1, 1}, 1);
    equal<unsigned char>(f.accumulated.get(), {1, 1, 0}, "microbatch union lost prior expert");
    equal<unsigned char>(f.current.get(), {0, 1, 0}, "current mask retained prior expert");
    equal<unsigned char>(f.first.get(), {0, 1, 0}, "new expert must overwrite");
    f.route({0, 1, 2, 2}, 2);
    equal<unsigned char>(f.first.get(), {0, 0, 0}, "repeat contributors must add");
    f.g.put({3, 4, 0, 12, std::numeric_limits<float>::quiet_NaN(), 0});
    require(launch_activity_multi_tensor_sqsum(f.sqsum.data, f.desc), "union norm enqueue failed");
    close(f.sqsum.get()[0], 169.0f, "optimizer norm lost a microbatch contributor");
    f.route({0, 0, 0, 0}, 0);
    equal<unsigned char>(f.accumulated.get(), {1, 1, 0}, "empty microbatch erased union");

    // Every invalid bank must fail before publishing any new expert, including
    // corruption discovered after an otherwise valid first segment.
    const std::array<std::vector<int>, 5> bad{{
        {1, 1, 2, 2}, {0, -1, 2, 2}, {0, 2, 1, 2}, {0, 1, 3, 2}, {0, 1, 1, 3}}};
    for (const auto& offsets : bad) {
        f.reset(); f.route({0, 1, 1, 1}, 1);
        f.route(offsets, 2);
        equal<int>(f.abort_issue.get(), {1}, "invalid routing did not latch abort");
        equal<unsigned char>(f.accumulated.get(), {1, 0, 0}, "invalid routing changed union");
        equal<unsigned char>(f.current.get(), {0, 0, 0}, "invalid routing retained current");
        equal<unsigned char>(f.first.get(), {0, 0, 0}, "invalid routing retained first-write");
        f.route({0, 0, 0, 1}, 1);
        equal<unsigned char>(f.accumulated.get(), {1, 0, 0}, "sticky abort accepted later routing");
        const auto before = f.w.get();
        f.adam(true);
        equal(f.w.get(), before, "routing abort did not gate Adam");
    }
    f.reset(); f.upstream_issue.put({7});
    f.route({0, 1, 1, 1}, 1, f.upstream_issue.data);
    equal<int>(f.abort_issue.get(), {1}, "upstream routing failure was ignored");
    equal<unsigned char>(f.accumulated.get(), {0, 0, 0}, "upstream failure published activity");
    f.reset(); f.route({0, 0, 0, 0}, 0);
    equal<unsigned char>(f.accumulated.get(), {0, 0, 0}, "empty routing became active");
    f.route({0, 0, 0, 1}, 8); // actual assignment count below capacity
    equal<int>(f.abort_issue.get(), {0}, "valid under-capacity routing rejected");
    require(launch_sparse_optimizer_activity_abort(f.activity), "explicit abort enqueue failed");
    equal<unsigned char>(f.accumulated.get(), {0, 0, 1}, "abort destroyed diagnostic union");
    equal<unsigned char>(f.current.get(), {0, 0, 0}, "abort did not clear current");
    f.reset();
    equal<unsigned char>(f.accumulated.get(), {0, 0, 0}, "reset failed to clear union");
    equal<int>(f.abort_issue.get(), {0}, "reset failed to clear abort");
}

void poisoned_inactive_and_clip_adam(bool chunked) {
    Fixture f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    f.g.put({3, 4, nan, nan, nan, nan});
    f.m.put({0, 0, nan, nan, 11, 12});
    f.v.put({0, 0, nan, nan, 13, 14});
    f.route({0, 1, 1, 1}, 1);
    f.finite(chunked);
    equal<int>(f.finite_issue.get(), {0}, "poisoned inactive gradient failed finite gate");
    require(launch_activity_multi_tensor_sqsum(f.sqsum.data, f.desc), "norm enqueue failed");
    close(f.sqsum.get()[0], 25.0f, "inactive poisoned gradient entered norm");
    const auto grad_before = f.g.get(), weight_before = f.w.get();
    const auto m_before = f.m.get(), v_before = f.v.get();
    require(launch_activity_chunked_norm_device_clip(f.total.data, f.partials.data,
                f.coefficient.data, f.finite_issue.data, f.desc, f.chunks.data, 3, 2.0f),
            "activity clip enqueue failed");
    close(static_cast<float>(f.total.get()[0]), 25.0f, "deterministic norm included inactive");
    const float scale = 2.0f / (5.0f + 1e-6f);
    const auto grad_after = f.g.get();
    close(grad_after[0], 3.0f * scale, "active gradient clip differs");
    close(grad_after[1], 4.0f * scale, "active gradient clip differs");
    require(std::memcmp(grad_after.data() + 2, grad_before.data() + 2, 4 * sizeof(float)) == 0,
            "clip wrote inactive poisoned gradient");
    // Stable finite gate includes the routing gate, latched by clip finalize.
    f.desc.contribution.abort_issue = f.finite_issue.data;
    f.adam(chunked);
    const auto weights = f.w.get(), moments1 = f.m.get(), moments2 = f.v.get();
    for (int i = 0; i < 2; ++i) {
        const float g = grad_after[i];
        const float expected_m = 0.5f * g, expected_v = 0.5f * g * g;
        const float expected_w = weight_before[i] - 0.1f * 0.2f * weight_before[i]
            - 0.1f * (expected_m / 0.5f) / (std::sqrt(expected_v / 0.5f) + 1e-8f);
        close(weights[i], expected_w, "Adam differs from independent scalar reference");
        close(moments1[i], expected_m, "Adam first moment differs");
        close(moments2[i], expected_v, "Adam second moment differs");
    }
    require(std::memcmp(weights.data() + 2, weight_before.data() + 2, 4 * sizeof(float)) == 0,
            "Adam decayed inactive weights");
    require(std::memcmp(moments1.data() + 2, m_before.data() + 2, 4 * sizeof(float)) == 0,
            "Adam wrote inactive first moments");
    require(std::memcmp(moments2.data() + 2, v_before.data() + 2, 4 * sizeof(float)) == 0,
            "Adam wrote inactive second moments");
    equal<int>(f.output_issue.get(), {0}, "inactive poison entered update finite output");

    // Next group: e0 retains storage and moments but has no contribution.
    f.reset(); f.route({0, 0, 1, 1}, 1);
    f.g.put({nan, nan, 0, 0, nan, nan});
    // e1 was logically absent: initialize only the newly active moment state.
    f.initialized.put({1, 0, 0});
    require(launch_activity_multi_tensor_initialize_moments(f.desc, f.initialized.data),
            "lazy moment init enqueue failed");
    f.adam(chunked);
    const auto next_weights = f.w.get(), next_m = f.m.get(), next_v = f.v.get();
    require(std::memcmp(next_weights.data(), weights.data(), 2 * sizeof(float)) == 0,
            "inactive-after-active expert changed weight");
    require(std::memcmp(next_m.data(), moments1.data(), 2 * sizeof(float)) == 0,
            "inactive-after-active expert changed m");
    require(std::memcmp(next_v.data(), moments2.data(), 2 * sizeof(float)) == 0,
            "inactive-after-active expert changed v");
    close(next_weights[2], weights[2] - 0.1f * 0.2f * weights[2], "zero-active did not apply decay");
    equal<unsigned char>(f.initialized.get(), {1, 1, 0}, "lazy presence initialized inactive expert");
}

void zero_active_lazy_and_finite_abort() {
    Fixture f;
    f.route({0, 0, 1, 1}, 1);  // zero grad, real contribution
    f.m.put({21, 22, 2, 2, 25, 26});
    f.v.put({31, 32, 4, 4, 35, 36});
    f.initialized.put({0, 1, 0});
    require(launch_activity_multi_tensor_initialize_moments(f.desc, f.initialized.data),
            "existing moment init enqueue failed");
    f.adam(false);
    const auto weights = f.w.get(), m = f.m.get(), v = f.v.get();
    close(m[2], 1.0f, "zero-active did not evolve existing m");
    close(v[2], 2.0f, "zero-active did not evolve existing v");
    close(weights[2], 3.0f - 0.1f * 0.2f * 3.0f - 0.1f * 2.0f / (2.0f + 1e-8f),
          "zero-active did not evolve weight");
    equal<unsigned char>(f.initialized.get(), {0, 1, 0}, "inactive logical moments became present");

    f.g.put({3, 4, std::numeric_limits<float>::infinity(), 0, 0, 0});
    f.finite(false);
    equal<int>(f.finite_issue.get(), {1}, "active nonfinite gradient passed gate");
    const auto grad_before = f.g.get();
    require(launch_activity_chunked_norm_device_clip(f.total.data, f.partials.data,
                f.coefficient.data, f.finite_issue.data, f.desc, f.chunks.data, 3, 2.0f),
            "nonfinite clip enqueue failed");
    equal(f.g.get(), grad_before, "failed finite gate changed gradients");
    f.desc.contribution.abort_issue = f.finite_issue.data;
    f.adam(true);
    equal(f.w.get(), weights, "failed finite gate changed weights");
    equal(f.m.get(), m, "failed finite gate changed m");
    equal(f.v.get(), v, "failed finite gate changed v");
    f.initialized.put({0, 0, 0});
    require(launch_activity_multi_tensor_initialize_moments(f.desc, f.initialized.data),
            "aborted lazy init enqueue failed");
    equal<unsigned char>(f.initialized.get(), {0, 0, 0}, "abort published lazy presence");
    equal(f.m.get(), m, "abort initialized moments");
}

void dense_legacy_null_entries_and_folded_lane() {
    Fixture f;
    f.desc.contribution = {nullptr, nullptr};
    f.finite(true);
    f.adam(true);
    const auto updated = f.w.get();
    close(updated[2], 3.0f - 0.1f * 0.2f * 3.0f, "null predicate table lost dense legacy decay");
    close(updated[4], 5.0f - 0.1f * 0.2f * 5.0f, "null predicate table lost dense legacy decay");

    Fixture partial;
    // A null predicate ENTRY represents a dense tensor in a mixed cohort.
    partial.predicates.put({partial.accumulated.data, nullptr, partial.accumulated.data + 2});
    partial.adam(false);
    close(partial.w.get()[2], 3.0f - 0.1f * 0.2f * 3.0f, "null predicate entry was treated inactive");
    close(partial.w.get()[0], 1.0f, "inactive mixed-cohort expert changed");

    Fixture folded;
    folded.route({0, 1, 1, 1}, 1);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    folded.g.put({3, 4, nan, nan, nan, nan});
    folded.lr.put({1, 1, 1}); // legacy lane interprets these as scales
    require(launch_activity_multi_tensor_sqsum(folded.sqsum.data, folded.desc), "folded norm failed");
    require(launch_activity_multi_tensor_adamw(folded.desc, folded.sqsum.data, 0.5f, 1.0f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.1f, 1e-8f, 0.2f, folded.output_issue.data),
            "folded Adam enqueue failed");
    const auto weights = folded.w.get();
    const float g0 = 3.0f * (0.5f * (1.0f / (2.5f + 1e-6f)));
    close(folded.m.get()[0], 0.5f * g0, "folded clip/accumulation differs");
    close(weights[0], 1.0f - 0.1f * 0.2f - 0.1f, "folded Adam differs");
    close(weights[2], 3.0f, "folded Adam decayed inactive");
    equal<int>(folded.output_issue.get(), {0}, "folded Adam read inactive poison");

    Fixture bad_norm;
    bad_norm.route({0, 1, 1, 1}, 1);
    bad_norm.sqsum.put({nan});
    const auto before_bad_norm = bad_norm.w.get();
    require(launch_activity_multi_tensor_adamw(bad_norm.desc, bad_norm.sqsum.data, 1.0f, 1.0f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.1f, 1e-8f, 0.2f, bad_norm.output_issue.data),
            "nonfinite folded norm enqueue failed");
    equal(bad_norm.w.get(), before_bad_norm, "nonfinite folded norm changed weights");
    equal<int>(bad_norm.output_issue.get(), {2}, "nonfinite folded norm failed to report output issue");

    Fixture nullable;
    nullable.route({0, 1, 1, 1}, 1);
    nullable.gradients.put({nullable.g.data, nullptr, nullptr});
    nullable.values.put({nullable.g.data, nullptr, nullptr});
    nullable.weights.put({nullable.w.data, nullptr, nullptr});
    nullable.moments1.put({nullable.m.data, nullptr, nullptr});
    nullable.moments2.put({nullable.v.data, nullptr, nullptr});
    nullable.finite(true);
    require(launch_activity_multi_tensor_sqsum(nullable.sqsum.data, nullable.desc), "null inactive norm failed");
    nullable.adam(false);
    close(nullable.sqsum.get()[0], 25.0f, "null inactive entry entered norm");
    equal<int>(nullable.output_issue.get(), {0}, "null inactive pointer was fetched");
}

void nonnegative_and_invalid_chunks() {
    Fixture f;
    f.route({0, 1, 1, 1}, 1);
    f.values.put({f.v.data, f.v.data + 2, f.v.data + 4});
    f.v.put({1, 2, -1, -2, -3, -4});
    require(launch_activity_multi_tensor_check_finite(f.finite_issue.data, f.values.data,
                f.nonnegative.data, f.desc, f.chunks.data, 3), "nonnegative finite enqueue failed");
    equal<int>(f.finite_issue.get(), {0}, "inactive negative moment failed finite gate");
    f.v.put({-1, 2, 1, 2, 3, 4});
    require(launch_activity_multi_tensor_check_finite(f.finite_issue.data, f.values.data,
                f.nonnegative.data, f.desc), "flat nonnegative finite enqueue failed");
    equal<int>(f.finite_issue.get(), {1}, "active negative moment passed finite gate");

    f.reset(); f.route({0, 1, 1, 1}, 1);
    f.chunks.put({{0, 2, 0}, {0, 2, 99}, {0, 2, 2}});
    const auto before = f.g.get();
    require(launch_activity_chunked_norm_device_clip(f.total.data, f.partials.data,
                f.coefficient.data, f.finite_issue.data, f.desc, f.chunks.data, 3, 1.0f),
            "invalid chunk clip enqueue failed");
    equal<int>(f.finite_issue.get(), {1}, "invalid chunk failed to latch gate");
    equal(f.g.get(), before, "invalid chunk changed gradients");
    require(!launch_activity_multi_tensor_adamw_update_deterministic(f.desc, nullptr, 3,
                0.5f, 0.5f, 0.5f, 0.5f, 1e-8f, 0.2f, f.output_issue.data),
            "host-invalid chunks accepted");
    f.desc.contribution.abort_issue = f.output_issue.data;
    require(!launch_activity_multi_tensor_adamw_update_deterministic(f.desc, nullptr, 0,
                0.5f, 0.5f, 0.5f, 0.5f, 1e-8f, 0.2f, f.output_issue.data),
            "mutable output alias accepted as stable pre-update gate");
}

void descriptor_preflight_and_scale() {
    Fixture f;
    f.route({0, 1, 1, 1}, 1);
    f.gradients.put({f.g.data, nullptr, nullptr});
    f.weights.put({f.w.data, nullptr, nullptr});
    f.moments1.put({f.m.data, nullptr, nullptr});
    f.moments2.put({f.v.data, nullptr, nullptr});
    require(launch_activity_multi_tensor_preflight(f.abort_issue.data, f.desc, f.chunks.data, 3),
            "valid descriptor preflight enqueue failed");
    equal<int>(f.abort_issue.get(), {0}, "preflight rejected null inactive storage");
    f.g.put({3, 4, 71, 72, 73, 74});
    require(launch_activity_multi_tensor_scale_gradients(f.desc, 0.25f), "activity scale enqueue failed");
    equal<float>(f.g.get(), {0.75f, 1.0f, 71, 72, 73, 74}, "scaling touched inactive gradients");

    const auto weights = f.w.get(), m = f.m.get(), v = f.v.get();
    f.gradients.put({nullptr, nullptr, nullptr});
    require(launch_activity_multi_tensor_preflight(f.abort_issue.data, f.desc),
            "invalid active pointer preflight enqueue failed");
    equal<int>(f.abort_issue.get(), {1}, "active missing storage passed preflight");
    f.adam(false);
    require(launch_activity_multi_tensor_initialize_moments(f.desc, f.initialized.data),
            "preflight-aborted lazy init enqueue failed");
    equal(f.w.get(), weights, "invalid active storage changed weights");
    equal(f.m.get(), m, "invalid active storage changed m");
    equal(f.v.get(), v, "invalid active storage changed v");
    equal<unsigned char>(f.initialized.get(), {0, 0, 0}, "invalid active storage published presence");

    Fixture chunks;
    chunks.route({0, 1, 1, 1}, 1);
    chunks.chunks.put({{0, 1, 0}, {0, 2, 1}, {0, 2, 2}}); // gap in tensor 0
    require(launch_activity_multi_tensor_preflight(chunks.abort_issue.data, chunks.desc, chunks.chunks.data, 3),
            "invalid chunk coverage preflight enqueue failed");
    equal<int>(chunks.abort_issue.get(), {1}, "incomplete chunk coverage passed preflight");
    const auto grad_before = chunks.g.get();
    require(launch_activity_multi_tensor_scale_gradients(chunks.desc, 0.25f), "aborted scale enqueue failed");
    equal(chunks.g.get(), grad_before, "preflight abort did not gate scaling");

    Fixture offsets;
    offsets.route({0, 1, 1, 1}, 1);
    offsets.tensor_offsets.put({0, 4, 2, 6});
    require(launch_activity_multi_tensor_preflight(offsets.abort_issue.data, offsets.desc),
            "invalid tensor offsets preflight enqueue failed");
    equal<int>(offsets.abort_issue.get(), {1}, "nonmonotonic tensor offsets passed preflight");

    Fixture learning_rate;
    learning_rate.route({0, 1, 1, 1}, 1);
    learning_rate.lr.put({std::numeric_limits<float>::infinity(), 0.1f, 0.1f});
    require(launch_activity_multi_tensor_preflight(learning_rate.abort_issue.data, learning_rate.desc),
            "invalid learning rate preflight enqueue failed");
    equal<int>(learning_rate.abort_issue.get(), {1}, "active invalid learning rate passed preflight");

    Fixture lazy;
    lazy.route({0, 1, 1, 1}, 1);
    lazy.m.put({std::numeric_limits<float>::quiet_NaN(), 0, 0, 0, 0, 0});
    lazy.values.put({lazy.m.data, lazy.m.data + 2, lazy.m.data + 4});
    require(launch_activity_multi_tensor_check_finite(lazy.finite_issue.data, lazy.values.data,
                nullptr, lazy.desc, lazy.chunks.data, 3, lazy.initialized.data),
            "lazy existing-moment finite enqueue failed");
    equal<int>(lazy.finite_issue.get(), {0}, "finite gate read logically absent active moment");
    lazy.initialized.put({1, 0, 0});
    require(launch_activity_multi_tensor_check_finite(lazy.finite_issue.data, lazy.values.data,
                nullptr, lazy.desc, nullptr, 0, lazy.initialized.data),
            "initialized existing-moment finite enqueue failed");
    equal<int>(lazy.finite_issue.get(), {1}, "finite gate skipped initialized active poisoned moment");
}
#endif
}  // namespace

int main() {
    return nsos::gpu_parity_test::run_parity("device_sparse_optimizer", [] {
#ifdef USE_CUDA
        accumulation_and_invalid_routing();
        poisoned_inactive_and_clip_adam(false);
        poisoned_inactive_and_clip_adam(true);
        zero_active_lazy_and_finite_abort();
        dense_legacy_null_entries_and_folded_lane();
        nonnegative_and_invalid_chunks();
        descriptor_preflight_and_scale();
#else
        throw std::runtime_error("device sparse optimizer requires a GPU backend");
#endif
    });
}
