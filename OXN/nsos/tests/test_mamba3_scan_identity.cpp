// Root builds/runs this CPU test after integrating the staged identity patch.
#include "jamba.h"
#include "trainer.h"
#include "tensor.h"
#include "gpu_backend.h"
#include "cuda/mamba3_layer_kernels.cuh"
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "gpu_execution.h"
#include "gpu_sparse_adam.h"
#include "nsos/determinism.h"
#include <cmath>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nsos;
namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class F> void rejected(F&& call,const char* message) {
    bool bad=false;try {call();}catch(const std::exception&){bad=true;}require(bad,message);
}
struct Env {
    std::string name, old;bool had=false;
    explicit Env(const char* key):name(key) {
        const char* value=std::getenv(key);had=value!=nullptr;if(value)old=value;
    }
    void set(const char* value) {
#ifdef _WIN32
        require(_putenv_s(name.c_str(),value?value:"")==0,"environment mutation");
#else
        require((value?setenv(name.c_str(),value,1):unsetenv(name.c_str()))==0,"environment mutation");
#endif
    }
    ~Env() {set(had?old.c_str():nullptr);}
};
ModelConfig config() {
    ModelConfig c;c.architecture_schema_version=3;c.num_layers=1;c.d_model=16;c.vocab_size=32;
    c.n_heads=2;c.n_kv_heads=1;c.attention_period=128;c.attention_slot=127;
    c.mamba3_enabled=true;c.mamba3_schema_version=1;c.mamba3_state_dim=128;
    c.mamba_expand=1;c.mamba_head_dim=8;c.mamba_n_groups=1;
    c.use_moe=false;c.use_ttt=false;c.use_kan=false;c.use_chrass=false;c.dropout=0;
    return c;
}
bool field(const std::vector<RuntimeExecutionIdentity::Field>& fields,const std::string& key) {
    for(const auto& f:fields)if(f.first==key)return true;return false;
}
std::string value(const std::vector<RuntimeExecutionIdentity::Field>& fields,const std::string& key) {
    for(const auto& f:fields)if(f.first==key)return f.second;return {};
}
void gpu_identity(Env& scan,Env& projection) {
#ifdef USE_CUDA
    require(gpu::select_preferred_device(nullptr,nullptr),"GPU capability unavailable; no successful skip");
    projection.set("exact_fp32");scan.set("dense_reference");
    JambaModel baseline_model(config(),Device::GPU);baseline_model.set_reference_path(true);
    Trainer baseline(&baseline_model,.002f);auto scheduler=baseline.phase_scheduler;
    baseline.warmup_steps=0;baseline.min_learning_rate_scale=1;
    scheduler.progressive_qat_enabled=false;baseline.configure_progressive_qat(scheduler);
    require(std::isfinite(baseline.train_step({1,2,3},{2,3,4})),"baseline GPU loss");
    const auto base_identity=baseline.execution_identity_fields();
    const auto path=std::filesystem::temp_directory_path()/"nsos-mamba3-scan-identity";
    const auto model_path=path.string()+".model",state_path=path.string()+".state";
    baseline_model.save(model_path);baseline.save_training_state(state_path,model_path);
    const auto v1_model=model_path+".flash-v1",v1_state=state_path+".flash-v1";
    const auto v2_model=model_path+".flash-v2",v2_state=state_path+".flash-v2";
    for(const char* mode:{"parallel_fp32_v1","flash_fp32_v1","flash_fp32_replay_lds_v2","flash_fp32_hierarchical_v1"}) {
        scan.set(mode);JambaModel target(config(),Device::GPU);target.load(model_path);
        Trainer resume(&target,.002f);resume.configure_progressive_qat(scheduler);
        const auto fields=resume.execution_identity_fields();
        require(value(fields,"mamba3.scan_provider")==mode,"wrong GPU scan identity");
        require(value(fields,"mamba3.scan_tile_tokens")=="32","missing scan width identity");
        require(fields!=base_identity,"scan identity aliased reference");
        require(value(fields,"mamba.scan_geometry")== (std::string(mode)=="flash_fp32_hierarchical_v1"?"mamba3_tile32_local_arity32_tree_128threads_4cells_v1":"mamba3_tile32_affine_prefix_128threads_4cells_v1"),"scan geometry still serial");
        if(std::string(mode)=="flash_fp32_v1"||std::string(mode)=="flash_fp32_replay_lds_v2") {
            require(value(fields,"mamba.history_layout")=="mamba3_tile32_boundaries_bh_pn_v1","Flash layout still dense");
            require(value(fields,"mamba3.history")== (std::string(mode)=="flash_fp32_replay_lds_v2"?"tile32_boundaries_t4_lds_explicit_halos_v2":"tile32_boundaries_replay_v1"),"Flash history identity");
            require(value(fields,"checkpoint.gradient_policy")== (std::string(mode)=="flash_fp32_replay_lds_v2"?"retain_mamba3_tile32_boundary_replay_lds_v2":"retain_mamba3_tile32_boundary_replay_v1"),"Flash checkpoint identity");
        }
        if(std::string(mode)=="flash_fp32_replay_lds_v2")
            require(value(fields,"mamba3.backward_replay")=="t4_p1_n128_stride129_explicit_scalar_seam_halos_v2","missing v2 replay geometry");
        rejected([&]{resume.load_training_state(state_path,model_path);},"reference sidecar accepted by optimized scan");
        require(resume.global_step_count==0&&!resume.optimizer_state_poisoned(),"mismatched resume mutated progress/poison");
        if(std::string(mode)=="flash_fp32_v1") {
            resume.warmup_steps=0;resume.min_learning_rate_scale=1;
            require(std::isfinite(resume.train_step({1,2,3},{2,3,4})),"Flash v1 checkpoint source loss");
            target.save(v1_model);resume.save_training_state(v1_state,v1_model);
        }
        if(std::string(mode)=="flash_fp32_replay_lds_v2") {
            rejected([&]{resume.load_training_state(v1_state,v1_model);},"Flash v1 sidecar accepted by v2 replay");
            require(resume.global_step_count==0&&!resume.optimizer_state_poisoned(),"v1-v2 mismatch changed progress/poison");
            resume.warmup_steps=0;resume.min_learning_rate_scale=1;
            require(std::isfinite(resume.train_step({1,2,3},{2,3,4})),"Flash v2 source loss");
            target.save(v2_model);resume.save_training_state(v2_state,v2_model);
        }
        if(std::string(mode)=="flash_fp32_hierarchical_v1") {
            require(value(fields,"mamba3.state_prefix")=="tile32_hs_arity32_hs_prefix_fmaf_parent_fixup_v1","hierarchy prefix identity");
            require(value(fields,"mamba3.state_suffix")=="tile32_hs_arity32_hs_suffix_fmaf_parent_fixup_v1","hierarchy suffix identity");
            require(value(fields,"mamba3.phase_order")=="serial_token_fp32_wrap_v1","hierarchy reordered phase");
            require(value(fields,"mamba3.hierarchy_arity")=="32","hierarchy arity identity");
            require(value(fields,"mamba3.hierarchy_scratch")=="owned_pair_tree_fp32_reused_after_forward_v1","hierarchy tape scratch identity");
            require(value(fields,"mamba3.history")=="hier32_tile32_boundaries_t4_lds_scalar_halos_v1","hierarchy history identity");
            require(value(fields,"checkpoint.gradient_policy")=="retain_mamba3_hier32_tile32_boundary_replay_lds_v1","hierarchy checkpoint identity");
            rejected([&]{resume.load_training_state(v1_state,v1_model);},"v1 sidecar accepted by hierarchy");
            rejected([&]{resume.load_training_state(v2_state,v2_model);},"v2 sidecar accepted by hierarchy");
            require(resume.global_step_count==0&&!resume.optimizer_state_poisoned(),"hierarchy identity reject mutated optimizer");
        }
        // WMMA projection identity remains orthogonal to the scan selection.
        for(const auto& arithmetic:std::vector<std::pair<const char*,const char*>>{
            {"rdna_bf16_v1","rdna3_wave32_tile32_bf16_operands_adjoint_fp32acc_controltail_fp32_v1"},
            {"rdna_fp16_v1","rdna3_wave32_tile32_fp16_operands_adjoint_fp32acc_controltail_fp32_v1"}}) {
            projection.set(arithmetic.first);Trainer mixed(&target,.002f);
            const auto combined=mixed.execution_identity_fields();
            require(value(combined,"mamba3.scan_provider")==mode,"WMMA lost scan identity");
            require(value(combined,"mamba3.projection_arithmetic")==arithmetic.second,
                "scan overwrote WMMA/control-tail identity");
        }
        projection.set("exact_fp32");
    }
    std::filesystem::remove(v1_model);std::filesystem::remove(v1_state);
    std::filesystem::remove(v2_model);std::filesystem::remove(v2_state);
    std::filesystem::remove(model_path);std::filesystem::remove(state_path);
#else
    throw std::runtime_error("GPU identity test requires GPU build");
#endif
}
// Root-only GPU gate: continuation within ONE hierarchical identity.
// No old/new-provider trajectory equality and no optimizer comparison.
#ifdef USE_CUDA
void hierarchical_checkpoint_roundtrip() {
    Env scan("NSOS_MAMBA3_GPU_PROVIDER"), projection("NSOS_MAMBA3_PROJECTION_PROVIDER");
    Env optimizer("NSOS_OPTIMIZER"), fused("NSOS_OPTIMIZER_FUSED_EPILOGUE");
    optimizer.set("adamw");fused.set("1");
    scan.set("flash_fp32_hierarchical_v1");projection.set("exact_fp32");
    nsos::set_strict_gpu_execution(true);
    nsos::determinism::set_deterministic_reductions(true);
    set_matmul_precision_mode(0);
    require(gpu::select_preferred_device(nullptr,nullptr),"hierarchy checkpoint requires GPU");
    std::string comparison_phase;
    auto tensor_equal=[&](const Tensor& left,const Tensor& right,const std::string& label) {
        require(left.shape==right.shape&&left.size==right.size,"checkpoint tensor shape/size");
        if(!left.size)return;
        const Tensor a=left.cpu(),b=right.cpu();
        for(int i=0;i<a.size;++i)
            require(std::isfinite(a.data()[i])&&std::isfinite(b.data()[i]),"checkpoint nonfinite tensor");
        if(std::memcmp(a.data(),b.data(),std::size_t(a.size)*sizeof(float))!=0) {
            for(int i=0;i<a.size;++i) {
                uint32_t lhs=0,rhs=0;
                std::memcpy(&lhs,a.data()+i,sizeof(lhs));std::memcpy(&rhs,b.data()+i,sizeof(rhs));
                if(lhs!=rhs) {
                    std::cerr<<"CHECKPOINT_MISMATCH phase="<<comparison_phase<<" tensor="<<label
                             <<" element="<<i<<" left="<<a.data()[i]<<" right="<<b.data()[i]
                             <<" left_bits="<<std::hex<<lhs<<" right_bits="<<rhs<<std::dec<<"\n";
                    break;
                }
            }
            throw std::runtime_error("hierarchy checkpoint tensor differs bitwise");
        }
    };
    for(const auto& geometry:std::vector<std::pair<int,int>>{{33,1},{1025,4}}) {
        const int S=geometry.first,R=geometry.second;
        auto c=config();c.mamba3_mimo=R>1;c.mamba3_mimo_rank=R;
        JambaModel source(c,Device::GPU);source.set_reference_path(true);
        Trainer uninterrupted(&source,.002f);auto schedule=uninterrupted.phase_scheduler;
        schedule.progressive_qat_enabled=false;uninterrupted.configure_progressive_qat(schedule);
        uninterrupted.warmup_steps=0;uninterrupted.min_learning_rate_scale=1;
        uninterrupted.total_training_steps=8;source.set_training_rng_sequence(20261002+R);
        std::vector<int> x(S),y(S);
        for(int i=0;i<S;++i) {x[i]=(i*7+R)%c.vocab_size;y[i]=(i*11+3)%c.vocab_size;}
        const auto identity=uninterrupted.execution_identity_fields();
        require(value(identity,"mamba3.scan_provider")=="flash_fp32_hierarchical_v1","checkpoint provider");
        require(value(identity,"optimizer.device_adam_update_arithmetic")==
            "serial_fp32_lr_times_mhat_before_denominator_divide_v2","checkpoint fused Adam arithmetic");
        require(std::isfinite(uninterrupted.train_step(x,y)),"checkpoint source step");
        require(uninterrupted.global_step_count==1&&uninterrupted.tokens_processed==S&&
                uninterrupted.tokens_committed==S,"checkpoint source commit/tokens");
        const auto dir=std::filesystem::temp_directory_path()/
            ("nsos-hier-checkpoint-"+std::to_string(S)+"-"+std::to_string(R)+"-"+
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(dir),"checkpoint output directory must be new");
        const auto model=(dir/"model.bin").string(),state=(dir/"state.bin").string();
        auto snapshot=uninterrupted.capture_checkpoint_snapshot();snapshot->write(model,state);
        rejected([&]{snapshot->write(model,state);},"hierarchical snapshot write twice");
        JambaModel target(c,Device::GPU);target.set_reference_path(true);target.load(model,true);
        Trainer resumed(&target,.02f);resumed.configure_progressive_qat(schedule);
        resumed.load_training_state(state,model);
        auto same=[&] {
            require(resumed.execution_identity_fields()==identity&&
                    uninterrupted.execution_identity_fields()==identity,"checkpoint identity changed");
            require(resumed.global_step_count==uninterrupted.global_step_count&&
                    resumed.tokens_processed==uninterrupted.tokens_processed&&
                    resumed.tokens_committed==uninterrupted.tokens_committed&&
                    resumed.pending_accumulation_microbatches==0&&
                    resumed.pending_accumulated_tokens==0&&
                    resumed.learning_rate==uninterrupted.learning_rate&&
                    resumed.learning_rate==.002f&&
                    !resumed.optimizer_state_poisoned(),"checkpoint progress/lr/owner");
            require(target.training_rng_sequence()==source.training_rng_sequence(),"checkpoint RNG sequence");
            const auto a=source.parameters(),b=target.parameters();
            require(a.size()==b.size()&&!a.empty(),"checkpoint parameter registry");
            auto moments=[&](const auto& m,const auto& n,const char* kind) {
                require(!m.empty()&&m.size()==n.size(),"checkpoint moment count");
                for(std::size_t i=0;i<a.size();++i) {
                    const auto u=m.find(a[i]),v=n.find(b[i]);
                    require((u==m.end())==(v==n.end()),"checkpoint moment owner membership");
                    if(u!=m.end())tensor_equal(u->second,v->second,std::string(kind)+":"+a[i]->name);
                }
            };
            for(std::size_t i=0;i<a.size();++i) {
                require(a[i]&&b[i]&&a[i]->data.get_device()==Device::GPU&&
                        b[i]->data.get_device()==Device::GPU,"checkpoint parameter residence");
                tensor_equal(a[i]->data,b[i]->data,"weight:"+a[i]->name);
            }
            require(uninterrupted.device_sparse_adam&&
                    (uninterrupted.global_step_count==1||resumed.device_sparse_adam),
                    "checkpoint live owner absent after continuation");
            // Host maps are checkpoint mirrors. The supported API binds each
            // owning lane and exports the authoritative logical moment cohort.
            // It rejects an unfinished group; no owner/lane check is bypassed.
            uninterrupted.synchronize_device_sparse_checkpoint();
            resumed.synchronize_device_sparse_checkpoint();
            moments(uninterrupted.m_state,resumed.m_state,"owner-m");
            moments(uninterrupted.v_state,resumed.v_state,"owner-v");
        };
        comparison_phase="restore S="+std::to_string(S)+" R="+std::to_string(R);
        same();
        for(int step=0;step<3;++step) {
            const float a=uninterrupted.train_step(x,y),b=resumed.train_step(x,y);
            require(std::isfinite(a)&&std::isfinite(b)&&std::memcmp(&a,&b,sizeof(float))==0,
                    "hierarchy checkpoint continuation loss differs bitwise");
            require(resumed.global_step_count==step+2&&resumed.tokens_committed==S*(step+2),
                    "hierarchy checkpoint continuation did not commit");
            comparison_phase="continuation "+std::to_string(step+2)+" S="+std::to_string(S)+" R="+std::to_string(R);
            same();
        }
        // Artifacts remain on failure; delete only the files created by this successful case.
        std::filesystem::remove(model);std::filesystem::remove(state);
        std::cout<<"PASS hierarchical checkpoint S="<<S<<" R="<<R
                 <<" source1+continuation3 all weights/moments/losses bitwise\n";
    }
}
#endif

}
int main(int argc,char** argv) {
    try {
        Env scan("NSOS_MAMBA3_GPU_PROVIDER"),projection("NSOS_MAMBA3_PROJECTION_PROVIDER");
        if(argc==2&&std::string(argv[1])=="--gpu") {
            gpu_identity(scan,projection);
#ifdef USE_CUDA
            hierarchical_checkpoint_roundtrip();
#endif
            std::cout<<"PASS GPU Mamba3 scan selection / mismatched sidecar rejection / WMMA orthogonal identity\n";return 0;
        }
        projection.set("exact_fp32");scan.set(nullptr);set_matmul_precision_mode(0);
        require(mamba3_block::gpu_provider_from_environment()==mamba3_block::GpuProvider::DenseReference,"default scan");
        JambaModel model(config(),Device::CPU);Trainer baseline(&model,.002f);
        const auto initial=baseline.execution_identity_fields();
        require(!field(initial,"mamba3.scan_provider")&&!field(initial,"mamba3.scan_tile_tokens"),"baseline identity changed");
        scan.set("dense_reference");Trainer explicit_reference(&model,.002f);
        require(explicit_reference.execution_identity_fields()==initial,"explicit/default reference identity differs");
        for(const char* mode:{"parallel_fp32_v1","flash_fp32_v1","flash_fp32_replay_lds_v2","flash_fp32_hierarchical_v1"}) {
            scan.set(mode);
            require(mamba3_block::gpu_provider_from_environment()!=mamba3_block::GpuProvider::DenseReference,"optimized selection aliased reference");
            Trainer cpu(&model,.002f);
            rejected([&]{cpu.execution_identity_fields();},"CPU accepted optimized GPU identity");
        }
        for(const char* mode:{"","parallel","flash","garbage"}) {
            scan.set(mode);
#ifdef _WIN32
            // _putenv_s("", "") removes a variable on Windows; default is valid.
            if(!*mode)continue;
#endif
            rejected([]{mamba3_block::gpu_provider_from_environment();},"unknown scan provider accepted");
            Trainer invalid(&model,.002f);
            rejected([&]{invalid.execution_identity_fields();},"unknown scan identity accepted");
        }
        scan.set("dense_reference");Trainer restored(&model,.002f);
        require(restored.execution_identity_fields()==initial,"negative selection polluted default identity");
        std::cout<<"PASS Mamba3 scan selection / CPU fail-closed / legacy reference identity\n";return 0;
    } catch(const std::exception& error) {std::cerr<<"FAIL "<<error.what()<<'\n';return 1;}
}
