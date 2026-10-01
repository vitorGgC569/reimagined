#include "gpu_sparse_adam.h"
#include "gpu_attention_training.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#ifdef USE_CUDA
#include "cuda/sparse_optimizer_activity.cuh"
#endif

namespace nsos {
#ifdef USE_CUDA
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool accepted) {
    if (!accepted) throw std::runtime_error("Device sparse Adam launch rejected");
}
template<class T> void upload(cuda_detail::DeviceBuffer<T>& target, const std::vector<T>& values) {
    if (!target.ensure(values.size())) throw std::bad_alloc();
    if (!values.empty()) {
        check(cudaMemcpy(target.get(), values.data(), sizeof(T)*values.size(), cudaMemcpyHostToDevice));
        record_gpu_transfer(Device::GPU, Device::CPU, sizeof(T)*values.size());
    }
}
template<class T> std::vector<T> download(const T* source, size_t count) {
    std::vector<T> out(count);
    check(cudaMemcpyAsync(out.data(), source, sizeof(T)*count, cudaMemcpyDeviceToHost, gpu::current_stream()));
    record_gpu_transfer(Device::CPU, Device::GPU, sizeof(T)*count);
    check(cudaStreamSynchronize(gpu::current_stream())); record_gpu_stream_synchronization();
    return out;
}
}
struct GpuSparseAdam::Impl {
    std::vector<GpuSparseAdamSlot> slots;
    std::vector<std::string> names;
    std::vector<Tensor> m, v;
    std::vector<uint64_t> versions;
    std::vector<float*> weight_storage, gradient_storage;
    std::vector<std::shared_ptr<GpuGradientActivity>> domains;
    std::function<void(int*)> merge_upstream_status;
    cuda_detail::DeviceBuffer<float*> w_, g_, m_, v_;
    cuda_detail::DeviceBuffer<const unsigned char*> predicates_;
    cuda_detail::DeviceBuffer<const int*> issues_;
    cuda_detail::DeviceBuffer<unsigned long long> offsets_;
    cuda_detail::DeviceBuffer<unsigned char> wd_, dense_, initialized_, backup_initialized_, nonnegative_, cache_valid_;
    cuda_detail::DeviceBuffer<uint64_t> versions_;
    cuda_detail::DeviceBuffer<NsosMultiTensorChunk> chunks_;
    cuda_detail::DeviceBuffer<float> learning_rates_, backup_, coefficient_, sqsum_;
    cuda_detail::DeviceBuffer<double> partials_, total_;
    cuda_detail::DeviceBuffer<int> status_;
    NsosActivityAwareOptimizerDesc desc{};
    int chunk_count = 0, device = -1;
    cudaStream_t stream = nullptr;
    bool prepared = false, poisoned = false;
    void lane() const {
        int current = -1; check(cudaGetDevice(&current));
        if (current != device || gpu::current_stream() != stream)
            throw std::logic_error("Device sparse Adam requires its owning lane");
        if (poisoned) throw std::logic_error("Device sparse Adam recovery failed; rebuild runtime from checkpoint");
    }
    void validate_storage() const {
        for (size_t i = 0; i < slots.size(); ++i) {
            auto* p = slots[i].parameter;
            if (p->data.raw_data() != weight_storage[i] || p->grad.raw_data() != gradient_storage[i] ||
                p->version != versions[i] || p->data.get_device() != Device::GPU ||
                (p->grad.size && (p->grad.get_device() != Device::GPU || p->grad.shape != p->data.shape ||
                 p->grad.size != m[i].size)) || m[i].shape != p->data.shape)
                throw std::logic_error("Sparse Adam parameter storage/version changed");
        }
    }
};
#else
struct GpuSparseAdam::Impl {};
#endif
GpuSparseAdam::GpuSparseAdam() : impl_(std::make_unique<Impl>()) {}
GpuSparseAdam::~GpuSparseAdam() = default;

void GpuSparseAdam::configure(const std::vector<GpuSparseAdamSlot>& slots,
    std::function<void(int*)> merge_upstream_status) try {
#ifdef USE_CUDA
    auto& s = *impl_;
    if (s.prepared || slots.empty() || slots.size() > size_t(std::numeric_limits<int>::max()))
        throw std::logic_error("Invalid/nested sparse Adam transaction");
    if (s.device < 0) { check(cudaGetDevice(&s.device)); s.stream = gpu::current_stream(); }
    s.lane();
    check(cudaStreamSynchronize(s.stream)); record_gpu_stream_synchronization();
    const bool first = s.slots.empty();
    if (!first && s.slots.size() != slots.size()) throw std::logic_error("Sparse Adam registry size changed");
    std::set<Parameter*> unique;
    std::set<std::string> names;
    std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
    for (size_t i = 0; i < slots.size(); ++i) {
        auto* p = slots[i].parameter;
        if (!p || !p->trainable || !p->data.size || p->data.get_device() != Device::GPU ||
            !std::isfinite(slots[i].learning_rate) || slots[i].learning_rate < 0 ||
            !unique.insert(p).second || p->name.empty() || !names.insert(p->name).second ||
            (!first && (s.slots[i].parameter != p || s.names[i] != p->name || s.m[i].shape != p->data.shape)))
            throw std::invalid_argument("Sparse Adam registry/shape/LR/name mismatch");
        if (p->grad.size && (p->grad.shape != p->data.shape || p->grad.get_device() != Device::GPU))
            throw std::invalid_argument("Sparse Adam gradient shape/device mismatch");
        const auto& binding = p->device_gradient_binding();
        if (binding.owner) binding.owner->assert_lane();
        // Capture dense_contributed before reservation; the caller supplies it
        // explicitly, never derive activity from allocated candidate storage.
        if (!p->grad.size && (binding.owner || slots[i].dense_contributed)) p->grad = Tensor::uninitialized(p->data.shape.dims, Device::GPU);
        for (const Tensor* t : {&p->data, &p->grad}) {
            if (!t->size) continue;
            const auto start = reinterpret_cast<uintptr_t>(t->raw_data());
            ranges.emplace_back(start, start + size_t(t->size)*sizeof(float));
        }
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
        if (ranges[i].first < ranges[i-1].second) throw std::logic_error("Sparse Adam writable storage overlaps");
    if (!first) for(size_t i=0;i<slots.size();++i)
        if(slots[i].parameter->data.raw_data()!=s.weight_storage[i] || slots[i].parameter->version!=s.versions[i])
            throw std::logic_error("Sparse Adam registry weight storage/version changed");
    if (first) {
        s.m.reserve(slots.size()); s.v.reserve(slots.size());
        for (const auto& slot : slots) {
            s.names.push_back(slot.parameter->name);
            s.m.push_back(Tensor::uninitialized(slot.parameter->data.shape.dims, Device::GPU));
            s.v.push_back(Tensor::uninitialized(slot.parameter->data.shape.dims, Device::GPU));
            s.versions.push_back(slot.parameter->version);
        }
        upload(s.initialized_, std::vector<unsigned char>(slots.size(),0));
        upload(s.versions_, s.versions);
    }
    s.slots = slots; s.domains.clear(); s.merge_upstream_status=std::move(merge_upstream_status);
    std::vector<float*> w, g, m, v;
    std::vector<float> lr;
    std::vector<unsigned char> wd, dense;
    std::vector<unsigned long long> offsets{0};
    std::vector<NsosMultiTensorChunk> chunks;
    const auto chunk_elements = optimizer_policy::deterministic_adamw_chunk_elements();
    for (size_t i = 0; i < slots.size(); ++i) {
        auto* p = slots[i].parameter;
        w.push_back(p->data.raw_data()); g.push_back(p->grad.raw_data());
        m.push_back(s.m[i].raw_data()); v.push_back(s.v[i].raw_data());
        lr.push_back(slots[i].learning_rate); wd.push_back(slots[i].weight_decay ? 1 : 0);
        dense.push_back(slots[i].dense_contributed ? 1 : 0);
        const auto count = static_cast<unsigned long long>(p->data.size);
        if (count > std::numeric_limits<unsigned long long>::max() - offsets.back()) throw std::overflow_error("Adam bank size");
        offsets.push_back(offsets.back()+count);
        for (unsigned long long begin = 0; begin < count; begin += chunk_elements)
            chunks.push_back({begin, static_cast<uint32_t>((std::min)(count-begin, static_cast<unsigned long long>(chunk_elements))), static_cast<uint32_t>(i)});
        const auto& binding = p->device_gradient_binding();
        if (binding.owner && std::find(s.domains.begin(),s.domains.end(),binding.owner)==s.domains.end()) s.domains.push_back(binding.owner);
    }
    if (chunks.size() > size_t(std::numeric_limits<int>::max()) ||
        offsets.back() > std::numeric_limits<size_t>::max()/(4*sizeof(float))) throw std::length_error("Sparse Adam scratch size");
    upload(s.w_,w); upload(s.g_,g); upload(s.m_,m); upload(s.v_,v);
    upload(s.learning_rates_,lr); upload(s.wd_,wd); upload(s.dense_,dense); upload(s.offsets_,offsets);
    upload(s.chunks_,chunks); upload(s.nonnegative_,std::vector<unsigned char>(slots.size(),1));
    upload(s.cache_valid_,std::vector<unsigned char>(slots.size(),1));
    std::vector<const unsigned char*> predicates;
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto& b = slots[i].parameter->device_gradient_binding();
        predicates.push_back(b.owner ? b.owner->predicate(b.expert) : s.dense_.get()+i);
    }
    upload(s.predicates_,predicates);
    std::vector<const int*> issues; for (const auto& domain : s.domains) issues.push_back(domain->issue());
    if (!issues.empty()) upload(s.issues_,issues);
    if (!s.status_.ensure(2) || !s.backup_.ensure(size_t(4*offsets.back())) ||
        !s.backup_initialized_.ensure(slots.size()) || !s.partials_.ensure(chunks.size()) ||
        !s.total_.ensure(1) || !s.coefficient_.ensure(1) || !s.sqsum_.ensure(1)) throw std::bad_alloc();
    s.desc = {s.w_.get(),s.g_.get(),s.m_.get(),s.v_.get(),s.offsets_.get(),s.wd_.get(),
        s.learning_rates_.get(),static_cast<int>(slots.size()),offsets.back(),{s.predicates_.get(),s.status_.get()}};
    s.weight_storage = std::move(w); s.gradient_storage = std::move(g);
    s.chunk_count = static_cast<int>(chunks.size()); s.prepared = true;
#else
    (void)slots; (void)merge_upstream_status; throw std::runtime_error("Device sparse Adam requires GPU build");
#endif
}

catch (...) {
#ifdef USE_CUDA
    impl_->poisoned=true;
#endif
    throw;
}

GpuSparseAdamResult GpuSparseAdam::step(const GpuSparseAdamOptions& o) {
#ifdef USE_CUDA
    auto& s=*impl_; s.lane();
    if (!s.prepared) throw std::logic_error("Sparse Adam requires configured group");
    if (o.accumulation_steps < 1 || !std::isfinite(o.max_norm) || o.max_norm <= 0 ||
        !std::isfinite(o.beta1) || o.beta1 < 0 || o.beta1 >= 1 ||
        !std::isfinite(o.beta2) || o.beta2 < 0 || o.beta2 >= 1 ||
        !std::isfinite(o.bc1) || o.bc1 <= 0 || o.bc1 > 1 || !std::isfinite(o.bc2) || o.bc2 <= 0 || o.bc2 > 1 ||
        !std::isfinite(o.eps) || o.eps <= 0 || !std::isfinite(o.weight_decay) || o.weight_decay < 0)
        throw std::invalid_argument("Invalid sparse Adam hyperparameters");
    s.validate_storage();
    bool snapshot_ready=false;
    try {
        check(cudaMemsetAsync(s.status_.get(),0,2*sizeof(int),s.stream));
        std::vector<Parameter*> status_parameters;
        status_parameters.reserve(s.slots.size());
        for (const auto& slot : s.slots) status_parameters.push_back(slot.parameter);
        attention_training::merge_status(status_parameters,s.status_.get());
        if(s.merge_upstream_status)s.merge_upstream_status(s.status_.get());
        require(launch_activity_merge_abort(s.status_.get(),s.issues_.get(),static_cast<int>(s.domains.size())));
        require(launch_activity_multi_tensor_preflight(s.status_.get(),s.desc,s.chunks_.get(),s.chunk_count));
        require(launch_activity_preflight_versions(s.desc,s.versions_.get(),s.status_.get()));
        auto finite=[&](float* const* values,const unsigned char* nonnegative,const unsigned char* initialized,int* status) {
            require(launch_activity_multi_tensor_check_finite(status,
                reinterpret_cast<const float* const*>(values),nonnegative,s.desc,s.chunks_.get(),s.chunk_count,initialized));
        };
        finite(s.desc.w,nullptr,nullptr,s.status_.get()); finite(s.desc.g,nullptr,nullptr,s.status_.get());
        finite(s.desc.m,nullptr,s.initialized_.get(),s.status_.get());
        finite(s.desc.v,s.nonnegative_.get(),s.initialized_.get(),s.status_.get());
        if (download(s.status_.get(),1)[0]) { abort(); return {}; }
        require(launch_activity_snapshot(s.desc,s.initialized_.get(),s.backup_initialized_.get(),s.backup_.get()));
        // Confirm snapshot completion before arming rollback against launch faults.
        check(cudaStreamSynchronize(s.stream)); record_gpu_stream_synchronization(); snapshot_ready=true;
        const float accumulation_scale=1.0f/static_cast<float>(o.accumulation_steps);
        if (o.deterministic) {
            require(launch_activity_multi_tensor_scale_gradients(s.desc,accumulation_scale));
            require(launch_activity_chunked_norm_device_clip(s.total_.get(),s.partials_.get(),s.coefficient_.get(),
                s.status_.get(),s.desc,s.chunks_.get(),s.chunk_count,o.max_norm));
        } else {
            check(cudaMemsetAsync(s.sqsum_.get(),0,sizeof(float),s.stream));
            require(launch_activity_multi_tensor_sqsum(s.sqsum_.get(),s.desc));
        }
        require(launch_activity_multi_tensor_initialize_moments(s.desc,s.initialized_.get()));
        if (o.deterministic)
            require(launch_activity_multi_tensor_adamw_update_deterministic(s.desc,s.chunks_.get(),s.chunk_count,
                o.beta1,o.beta2,o.bc1,o.bc2,o.eps,o.weight_decay,s.status_.get()+1));
        else
            require(launch_activity_multi_tensor_adamw(s.desc,s.sqsum_.get(),accumulation_scale,o.max_norm,
                o.beta1,o.beta2,o.bc1,o.bc2,1.0f,o.eps,o.weight_decay,s.status_.get()+1));
        finite(s.desc.w,nullptr,nullptr,s.status_.get()+1);
        finite(s.desc.m,nullptr,s.initialized_.get(),s.status_.get()+1);
        finite(s.desc.v,s.nonnegative_.get(),s.initialized_.get(),s.status_.get()+1);
        const auto result=download(s.status_.get(),2);
        if (result[0] || result[1]) {
            require(launch_activity_restore(s.desc,s.initialized_.get(),s.backup_initialized_.get(),s.backup_.get()));
            check(cudaStreamSynchronize(s.stream)); record_gpu_stream_synchronization(); snapshot_ready=false;
            abort(); return {};
        }
        double norm=0;
        if (o.deterministic) norm=std::sqrt(download(s.total_.get(),1)[0]);
        else norm=std::sqrt(static_cast<double>(download(s.sqsum_.get(),1)[0]))*accumulation_scale;
        require(launch_activity_publish_versions(s.desc,s.versions_.get(),s.cache_valid_.get(),s.status_.get()+1));
        const auto versions=download(s.versions_.get(),s.slots.size());
        // Ordered compatibility mirror for existing host-version caches. It is
        // never used to construct a host optimizer cohort or lazy moment state.
        for (size_t i=0;i<versions.size();++i) s.slots[i].parameter->version=versions[i];
        s.versions=versions; s.prepared=false;
        return {true,norm};
    } catch (...) {
        try {
            if (snapshot_ready) {
                require(launch_activity_restore(s.desc,s.initialized_.get(),s.backup_initialized_.get(),s.backup_.get()));
                check(cudaStreamSynchronize(s.stream)); record_gpu_stream_synchronization();
                upload(s.versions_,s.versions);
            }
            abort();
        } catch (...) { s.poisoned=true; }
        throw;
    }
#else
    (void)o; throw std::runtime_error("Device sparse Adam requires GPU build");
#endif
}
void GpuSparseAdam::abort() {
#ifdef USE_CUDA
    auto& s=*impl_; s.lane();
    for (const auto& domain:s.domains) domain->abort();
    s.prepared=false;
#endif
}
std::vector<GpuSparseAdamState> GpuSparseAdam::snapshot() const {
#ifdef USE_CUDA
    auto& s=*impl_; s.lane();
    if (s.prepared) throw std::logic_error("Snapshot requires a completed device optimizer boundary");
    const auto initialized=download(s.initialized_.get(),s.slots.size());
    std::vector<GpuSparseAdamState> states;
    for (size_t i=0;i<s.slots.size();++i) {
        const bool present=initialized[i]!=0;
        states.push_back({s.slots[i].parameter->name,s.versions[i],present,
            present?s.m[i]:Tensor(),present?s.v[i]:Tensor()});
    }
    return states;
#else
    throw std::runtime_error("Device sparse Adam requires GPU build");
#endif
}
void GpuSparseAdam::restore(const std::vector<GpuSparseAdamState>& states) {
#ifdef USE_CUDA
    auto& s=*impl_; s.lane();
    if (states.size()!=s.slots.size()) throw std::logic_error("Restore requires matching registry");
    // Validate/stage the complete checkpoint before publishing any state.
    std::vector<Tensor> m,v; std::vector<unsigned char> initialized;
    std::vector<uint64_t> versions;
    for (size_t i=0;i<states.size();++i) {
        const auto& state=states[i]; auto* p=s.slots[i].parameter;
        if (state.name!=p->name || !state.version ||
            (!state.initialized && (state.m.size || state.v.size)) ||
            (state.initialized && (state.m.shape!=p->data.shape || state.v.shape!=p->data.shape)))
            throw std::invalid_argument("Sparse Adam checkpoint metadata mismatch");
        if (state.initialized) {
            const auto mh=state.m.to(Device::CPU), vh=state.v.to(Device::CPU);
            for (size_t j=0;j<size_t(mh.size);++j)
                if (!std::isfinite(mh.data()[j]) || !std::isfinite(vh.data()[j]) || vh.data()[j]<0)
                    throw std::invalid_argument("Nonfinite/negative sparse Adam checkpoint moments");
            m.push_back(mh.to(Device::GPU)); v.push_back(vh.to(Device::GPU));
        } else {
            m.push_back(Tensor::uninitialized(p->data.shape.dims,Device::GPU));
            v.push_back(Tensor::uninitialized(p->data.shape.dims,Device::GPU));
        }
        initialized.push_back(state.initialized?1:0); versions.push_back(state.version);
    }
    check(cudaStreamSynchronize(s.stream)); record_gpu_stream_synchronization();
    std::vector<float*> mptr,vptr;
    for (size_t i=0;i<m.size();++i) {mptr.push_back(m[i].raw_data()); vptr.push_back(v[i].raw_data());}
    // All potentially failing allocation/staging happened above; a transfer
    // failure poisons this owner instead of leaving a usable partial restore.
    try {
        upload(s.m_,mptr); upload(s.v_,vptr); upload(s.initialized_,initialized); upload(s.versions_,versions);
        s.m=std::move(m);s.v=std::move(v);s.versions=versions;
        for(size_t i=0;i<versions.size();++i)s.slots[i].parameter->version=versions[i];
    } catch (...) {s.poisoned=true;throw;}
#else
    (void)states; throw std::runtime_error("Device sparse Adam requires GPU build");
#endif
}
} // namespace nsos
