#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "muon_math.h"
#include "optimizer_runtime_policy.h"
#include "muon_oracle_fixture.h"
#include <cstdlib>
#include <iostream>

namespace {
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
template<class F> void rejects(F&& f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"invalid input accepted");}
void env(const char* name,const char* value){
#ifdef _WIN32
    if(_putenv_s(name,value))throw std::runtime_error("environment write failed");
#else
    if(setenv(name,value,1))throw std::runtime_error("environment write failed");
#endif
}
struct EnvGuard {std::string name,value;bool exists;EnvGuard(const char* n):name(n),exists(std::getenv(n)!=nullptr){if(exists)value=std::getenv(n);}~EnvGuard(){
#ifdef _WIN32
    _putenv_s(name.c_str(),exists?value.c_str():"");
#else
    if(exists)setenv(name.c_str(),value.c_str(),1);else unsetenv(name.c_str());
#endif
}};
}
int main()try {
    double max_error=0;
    for(const auto& fixture:kMuonOracle) {
        auto momentum=fixture.incoming;
        const auto result=nsos::muon::direction(fixture.gradient,momentum,fixture.rows,fixture.cols);
        for(size_t i=0;i<result.size();++i) {
            const double error=std::abs(double(result[i])-fixture.direction[i]);max_error=std::max(max_error,error);
            require(error<4e-5,"pinned primary FP32-adapter direction differs");
            require(std::abs(momentum[i]-fixture.momentum[i])<2e-7,"pinned primary momentum differs");
        }
    }
    std::vector<float> zero(6),momentum(6);
    require(nsos::muon::direction(zero,momentum,3,2)==zero,"zero matrix is not exact zero");
    auto before=momentum;auto bad=zero;bad[3]=std::numeric_limits<float>::quiet_NaN();
    rejects([&]{nsos::muon::direction(bad,momentum,2,3);});require(momentum==before,"nonfinite input published momentum");
    rejects([&]{nsos::muon::direction(zero,momentum,3,3);});require(momentum==before,"shape rejection published momentum");
    bad.assign(6,std::numeric_limits<float>::max());
    momentum=bad;before=momentum;
    rejects([&]{nsos::muon::direction(bad,momentum,3,2);});require(momentum==before,"overflow published momentum");
    require(nsos::muon::hidden_matrix("layers.0.mamba3.in_proj.weight",{6,3}),"hidden projection rejected");
    require(nsos::muon::hidden_matrix("layers.0.experts.2.up.weight",{6,3}),"hidden expert rejected");
    for(const auto& name:{"embedding.weight","lm_head.weight","layers.0.router.gate.weight","layers.0.mamba3.dt_bias","layers.0.mamba3.D"})
        require(!nsos::muon::hidden_matrix(name,{6,3}),"auxiliary parameter classified as Muon");
    require(!nsos::muon::hidden_matrix("layers.0.norm.weight",{6}),"vector classified as Muon");
    EnvGuard optimizer("NSOS_OPTIMIZER"),lr("NSOS_MUON_LR"),epilogue("NSOS_OPTIMIZER_FUSED_EPILOGUE"),moe("NSOS_MOE_DEVICE_ADAM");
    env("NSOS_OPTIMIZER","adamw");env("NSOS_MOE_DEVICE_ADAM","0");env("NSOS_OPTIMIZER_FUSED_EPILOGUE","0");require(!nsos::optimizer_policy::device_sparse_adam_enabled(),"default algorithm changed");
    env("NSOS_OPTIMIZER_FUSED_EPILOGUE","1");require(nsos::optimizer_policy::device_sparse_adam_enabled() && nsos::optimizer_policy::dense_device_optimizer_enabled(),"fused dense Adam lacks device transaction owner");
    env("NSOS_OPTIMIZER_FUSED_EPILOGUE","0");env("NSOS_MOE_DEVICE_ADAM","1");require(nsos::optimizer_policy::device_sparse_adam_enabled() && !nsos::optimizer_policy::dense_device_optimizer_enabled(),"legacy MoE opt-in silently became a dense policy");env("NSOS_MOE_DEVICE_ADAM","0");
    env("NSOS_OPTIMIZER","muon_ns5_fp32_v1");require(nsos::optimizer_policy::device_sparse_adam_enabled(),"Muon lacks device owner");
    env("NSOS_MUON_LR","");rejects([]{nsos::optimizer_policy::muon_learning_rate();});
    for(const char* value:{"0","-0.2","inf","nan","0.02garbage"," 0.02","1e999","1e-999"}) {env("NSOS_MUON_LR",value);rejects([]{nsos::optimizer_policy::muon_learning_rate();});}
    env("NSOS_MUON_LR","0.02");require(nsos::optimizer_policy::muon_learning_rate()==.02f,"explicit LR differs");
    const auto identity=nsos::optimizer_policy::muon_learning_rate_identity();env("NSOS_MUON_LR","0.020000001");
    require(identity!=nsos::optimizer_policy::muon_learning_rate_identity(),"LR identity lost FP32 bits");
    env("NSOS_OPTIMIZER","muon");rejects([]{nsos::optimizer_policy::muon_enabled();});
    env("NSOS_OPTIMIZER_FUSED_EPILOGUE","yes");rejects([]{nsos::optimizer_policy::fused_optimizer_epilogue_enabled();});
    std::cout<<"Muon: "<<kMuonOracle.size()<<" pinned primary FP32-adapter cases, max_error="<<max_error<<"; zero, rectangular, three-step momentum, transactional finite/shape and policy checks passed\n";
    return 0;
} catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
