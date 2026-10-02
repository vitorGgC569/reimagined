// Root builds/runs this CPU test after integrating the staged identity patch.
#include "jamba.h"
#include "trainer.h"
#include "tensor.h"
#include "gpu_backend.h"
#include "cuda/mamba3_layer_kernels.cuh"
#include <cstdlib>
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
    for(const char* mode:{"parallel_fp32_v1","flash_fp32_v1","flash_fp32_replay_lds_v2"}) {
        scan.set(mode);JambaModel target(config(),Device::GPU);target.load(model_path);
        Trainer resume(&target,.002f);resume.configure_progressive_qat(scheduler);
        const auto fields=resume.execution_identity_fields();
        require(value(fields,"mamba3.scan_provider")==mode,"wrong GPU scan identity");
        require(value(fields,"mamba3.scan_tile_tokens")=="32","missing scan width identity");
        require(fields!=base_identity,"scan identity aliased reference");
        require(value(fields,"mamba.scan_geometry")=="mamba3_tile32_affine_prefix_128threads_4cells_v1","scan geometry still serial");
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
    std::filesystem::remove(model_path);std::filesystem::remove(state_path);
#else
    throw std::runtime_error("GPU identity test requires GPU build");
#endif
}
}
int main(int argc,char** argv) {
    try {
        Env scan("NSOS_MAMBA3_GPU_PROVIDER"),projection("NSOS_MAMBA3_PROJECTION_PROVIDER");
        if(argc==2&&std::string(argv[1])=="--gpu") {
            gpu_identity(scan,projection);
            std::cout<<"PASS GPU Mamba3 scan selection / mismatched sidecar rejection / WMMA orthogonal identity\n";return 0;
        }
        projection.set("exact_fp32");scan.set(nullptr);set_matmul_precision_mode(0);
        require(mamba3_block::gpu_provider_from_environment()==mamba3_block::GpuProvider::DenseReference,"default scan");
        JambaModel model(config(),Device::CPU);Trainer baseline(&model,.002f);
        const auto initial=baseline.execution_identity_fields();
        require(!field(initial,"mamba3.scan_provider")&&!field(initial,"mamba3.scan_tile_tokens"),"baseline identity changed");
        scan.set("dense_reference");Trainer explicit_reference(&model,.002f);
        require(explicit_reference.execution_identity_fields()==initial,"explicit/default reference identity differs");
        for(const char* mode:{"parallel_fp32_v1","flash_fp32_v1","flash_fp32_replay_lds_v2"}) {
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
